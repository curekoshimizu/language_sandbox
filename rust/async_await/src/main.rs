use std::io;
use std::time::Instant;
use tokio::time;

async fn hello() -> String {
    // like "await asyncio.sleep(1)" in python
    time::sleep(time::Duration::from_secs(1)).await;

    "hello".to_string()
}

async fn sleep_msec(sleep_time: u64) -> u64 {
    // like "await asyncio.sleep(1)" in python
    time::sleep(time::Duration::from_millis(sleep_time)).await;

    sleep_time
}

// asyncio.Lock --> Mutex?
// asyncio.create_task
// run_in_executor

fn main() -> Result<(), io::Error> {
    // like. asyncio.run(coro()) in python
    tokio::runtime::Runtime::new()?.block_on(async_func());

    async_main_macro();

    Ok(())
}

async fn async_func() {
    let greeting: String = hello().await;
    println!("{}", greeting);

    // like. "await asyncio.wait_for(sleep_msec(), timeout=0.2)" in python
    let res = tokio::time::timeout(time::Duration::from_millis(200), sleep_msec(1000)).await;
    assert!(res.is_err());

    // like. asyncio.gather(coro1(), coro2()) in python
    let start = Instant::now();
    let ret = tokio::join!(sleep_msec(1000), sleep_msec(500), sleep_msec(700),);
    let elapsed_time = start.elapsed().as_millis();
    println!(">>> {:?}. elapsed_time {:?} [msec]", ret, &elapsed_time);

    let world = async {
        println!("hello, async block");
    };
    println!("hoge");
    world.await;
}

#[tokio::main]
async fn async_main_macro() {
    sleep_msec(1000).await;
}
