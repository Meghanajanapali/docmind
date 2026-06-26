import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set.")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # test connection before using it
    pool_recycle=300,        # recycle connections every 5 minutes
    pool_size=5,
    max_overflow=10
)

SessionLocal = sessionmaker(bind=engine)