
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "netflix_titles.csv"
PUBLIC_DIR = BASE_DIR / "public"
HTML_PATH = PUBLIC_DIR / "index.html"


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

    # Clean column names
    df.columns = df.columns.str.strip()

    # Clean text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Convert release year to numeric
    df["release_year"] = pd.to_numeric(
        df["release_year"], errors="coerce"
    )

    # Convert date_added to datetime
    df["date_added"] = pd.to_datetime(
        df["date_added"], errors="coerce"
    )

    # Replace pandas NaN/NaT with JSON-compatible None
    df = df.astype(object).where(pd.notna(df), None)

    return df


def get_dataset():
    try:
        return load_dataset()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load Netflix dataset: {exc}",
        )


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

def filter_dataset(
    df,
    content_type: Optional[str] = None,
    release_year: Optional[int] = None,
    search: Optional[str] = None,
):
    if content_type:
        df = df[df["type"] == content_type]

    if release_year:
        df = df[df["release_year"] == release_year]

    if search:
        df = df[
            df["title"]
            .fillna("")
            .str.contains(search, case=False, regex=False)
        ]

    return df


# --------------------------------------------------
# FRONTEND DASHBOARD
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    if not HTML_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Dashboard HTML not found: {HTML_PATH}",
        )

    return FileResponse(
        str(HTML_PATH),
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
        "rows": len(df),
        "csv_exists": DATA_PATH.exists(),
    }


# --------------------------------------------------
# FILTER OPTIONS
# --------------------------------------------------

@app.get("/filters")
def filters():
    df = get_dataset()

    years = sorted(
        [
            int(year)
            for year in df["release_year"].dropna().unique()
        ],
        reverse=True,
    )

    types = sorted(
        df["type"].dropna().unique().tolist()
    )

    countries = sorted(
        {
            country.strip()
            for value in df["country"].dropna()
            for country in str(value).split(",")
            if country.strip()
        }
    )

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

    movie_count = int((df["type"] == "Movie").sum())
    show_count = int((df["type"] == "TV Show").sum())

    countries = {
        country.strip()
        for value in df["country"].dropna()
        for country in str(value).split(",")
        if country.strip()
    }

    return {
        "total_titles": int(len(df)),
        "total_movies": movie_count,
        "total_tv_shows": show_count,
        "total_countries": len(countries),
        "first_release_year": (
            int(df["release_year"].min())
            if df["release_year"].notna().any()
            else None
        ),
        "last_release_year": (
            int(df["release_year"].max())
            if df["release_year"].notna().any()
            else None
        ),
    }


# --------------------------------------------------
# CHART DATA
# --------------------------------------------------

@app.get("/charts")
def charts():
    df = get_dataset()

    # Movie vs TV Show
    type_counts = (
        df["type"].value_counts().to_dict()
    )

    # Top genres
    genre_counts = {}

    for value in df["listed_in"].dropna():
        for genre in str(value).split(","):
            genre = genre.strip()

            if genre:
                genre_counts[genre] = (
                    genre_counts.get(genre, 0) + 1
                )

    top_genres = sorted(
        genre_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]

    # Titles by release year
    year_counts = (
        df["release_year"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
    )

    return {
        "type_counts": {
            str(key): int(value)
            for key, value in type_counts.items()
        },
        "top_genres": [
            {"genre": genre, "count": count}
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

@app.get("/titles")
def titles(
    limit: int = Query(default=100, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
    content_type: Optional[str] = None,
    release_year: Optional[int] = None,
    search: Optional[str] = None,
):
    df = get_dataset()

    if content_type and content_type not in [
        "Movie",
        "TV Show",
    ]:
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

    # Select table columns
    columns = [
        "show_id",
        "type",
        "title",
        "director",
        "country",
        "date_added",
        "release_year",
        "rating",
        "duration",
        "listed_in",
        "description",
    ]

    df = df[columns].iloc[offset:offset + limit]

    # Convert dates to strings for JSON
    df["date_added"] = df["date_added"].map(
        lambda value: (
            value.strftime("%B %d, %Y")
            if pd.notna(value)
            else None
        )
    )

    # Convert NaN/NaT to None
    df = df.astype(object).where(pd.notna(df), None)

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "items": df.to_dict(orient="records"),
    }


# --------------------------------------------------
# RUN LOCALLY
# --------------------------------------------------
# Start using:
# python -m uvicorn api.index:app --reload