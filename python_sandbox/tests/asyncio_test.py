import asyncio
import time


async def non_blocking() -> None:
    await asyncio.sleep(0.5)


async def blocking() -> None:
    time.sleep(0.8)  # warning error will appear


def test_slow_callback_duration() -> None:
    async def main() -> None:
        loop = asyncio.get_event_loop()
        loop.slow_callback_duration = 0.2  # default threshold is 0.1 sec

        await asyncio.gather(non_blocking(), blocking())

    asyncio.run(main(), debug=True)
