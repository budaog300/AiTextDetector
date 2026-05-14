from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingDB(BaseSettings):
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str

    model_config = SettingsConfigDict(env_file=".env.db", extra="ignore")

    @property
    def get_db_data(self) -> str:
        return f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


db_settings = SettingDB()
