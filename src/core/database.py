from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import text, func, UUID
from datetime import datetime
import uuid

from src.core.config import db_settings

DATABASE_URL = db_settings.get_db_data

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    __abstract__ = True


class Default(Base):
    __abstract__ = True

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), server_default=text("gen_random_uuid()"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(), server_default=text("TIMEZONE('utc', now())")
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        server_default=text("TIMEZONE('utc', now())"),
        onupdate=text("TIMEZONE('utc', now())"),
    )


async def get_db():
    async with async_session() as session:
        yield session
