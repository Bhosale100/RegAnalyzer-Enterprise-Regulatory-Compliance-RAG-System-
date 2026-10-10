import asyncio

# IMPORTANT: must be set before asyncio.run()
asyncio.set_event_loop_policy(
    asyncio.WindowsSelectorEventLoopPolicy()
)

from app.config import get_settings
from sqlalchemy.ext.asyncio import create_async_engine


async def main():
    settings = get_settings()

    engine = create_async_engine(
        settings.postgres_dsn_async
    )

    async with engine.connect():
        print("DATABASE CONNECTION SUCCESS")

    await engine.dispose()


asyncio.run(main())
