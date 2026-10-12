
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "netflix_titles.csv"
PUBLIC_DIR = BASE_DIR / "public"

HTML_PATH = PUBLIC_DIR / "index.html"
DASHBOARD_PATH = PUBLIC_DIR / "dashboard.html"


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Netflix Movies & TV Shows Analysis API",
    description="Netflix dataset analytics dashboard API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Netflix CSV file not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required_columns = [
        "show_id", "type", "title", "director",
        "country", "date_added", "release_year",
        "rating", "duration", "listed_in", "description"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    df["release_year"] = pd.to_numeric(
        df["release_year"], errors="coerce"
    )

    df["date_added"] = pd.to_datetime(
        df["date_added"], errors="coerce"
    )

    return df


def get_dataset():
    try:
        return load_dataset()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load Netflix dataset: {exc}",
        )


def json_safe(df):
    return df.astype(object).where(pd.notna(df), None)


# --------------------------------------------------
# SERVE HOME PAGE
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    if not HTML_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Home page not found: {HTML_PATH}",
        )

    return FileResponse(str(HTML_PATH), media_type="text/html")


# --------------------------------------------------
# SERVE DASHBOARD PAGE
# --------------------------------------------------

@app.get("/dashboard.html", include_in_schema=False)
def dashboard():
    if not DASHBOARD_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dashboard HTML not found: {DASHBOARD_PATH}",
        )

    return FileResponse(
        str(DASHBOARD_PATH),
        media_type="text/html",
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    df = get_dataset()

    return {
        "status": "ok",
        "rows": int(len(df)),
        "csv_exists": DATA_PATH.exists(),
    }


# --------------------------------------------------
# FILTER OPTIONS
# --------------------------------------------------

@app.get("/filters")
def filters():
    df = get_dataset()

    years = sorted(
        [int(y) for y in df["release_year"].dropna().unique()],
        reverse=True,
    )

    types = sorted(df["type"].dropna().unique().tolist())

    countries = sorted({
        country.strip()
        for value in df["country"].dropna()
        for country in str(value).split(",")
        if country.strip()
    })

    return {
        "types": types,
        "years": years,
        "countries": countries,
    }


# --------------------------------------------------
# SUMMARY STATISTICS
# --------------------------------------------------

@app.get("/summary")
def summary():
    df = get_dataset()

    countries = {
        country.strip()
        for value in df["country"].dropna()
        for country in str(value).split(",")
        if country.strip()
    }

    years = df["release_year"].dropna()

    return {
        "total_titles": int(len(df)),
        "total_movies": int((df["type"] == "Movie").sum()),
        "total_tv_shows": int((df["type"] == "TV Show").sum()),
        "total_countries": len(countries),
        "first_release_year": int(years.min()) if not years.empty else None,
        "last_release_year": int(years.max()) if not years.empty else None,
    }


# --------------------------------------------------
# CHART DATA
# --------------------------------------------------

@app.get("/charts")
def charts():
    df = get_dataset()

    type_counts = df["type"].value_counts().to_dict()

    genre_counts = {}
    for value in df["listed_in"].dropna():
        for genre in str(value).split(","):
            genre = genre.strip()
            if genre:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

    top_genres = sorted(
        genre_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    year_counts = (
        df["release_year"].dropna().astype(int).value_counts().sort_index()
    )

    return {
        "type_counts": {
            str(key): int(value)
            for key, value in type_counts.items()
        },
        "top_genres": [
            {"genre": genre, "count": int(count)}
            for genre, count in top_genres
        ],
        "release_years": [
            {"year": int(year), "count": int(count)}
            for year, count in year_counts.items()
        ],
    }


# --------------------------------------------------
# TITLES TABLE + SEARCH + FILTERS
# --------------------------------------------------

def filter_dataset(
    df,
    content_type: Optional[str] = None,
    release_year: Optional[int] = None,
    search: Optional[str] = None,
):
    if content_type:
        df = df[df["type"] == content_type]

    if release_year is not None:
        df = df[df["release_year"] == release_year]

    if search:
        df = df[
            df["title"].fillna("").str.contains(
                search, case=False, regex=False
            )
        ]

    return df


@app.get("/titles")
def titles(
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    content_type: Optional[str] = None,
    release_year: Optional[int] = None,
    search: Optional[str] = None,
):
    df = get_dataset()

    if content_type and content_type not in ["Movie", "TV Show"]:
        raise HTTPException(
            status_code=400,
            detail="content_type must be Movie or TV Show",
        )

    df = filter_dataset(
        df,
        content_type=content_type,
        release_year=release_year,
        search=search,
    )

    total = len(df)

    columns = [
        "show_id", "type", "title", "director", "country",
        "date_added", "release_year", "rating", "duration",
        "listed_in", "description",
    ]

    df = df[columns].iloc[offset:offset + limit].copy()

    df["date_added"] = pd.to_datetime(
        df["date_added"], errors="coerce"
    ).map(
        lambda value: value.strftime("%B %d, %Y")
        if pd.notna(value) else None
    )

    df["release_year"] = pd.to_numeric(
        df["release_year"], errors="coerce"
    )

    df = json_safe(df)

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "items": df.to_dict(orient="records"),
    }
