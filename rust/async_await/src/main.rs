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

// TODO:
// asyncio.Lock --> Mutex?
// asyncio.generator --> tokio_streams?
// rx, channel
// web server

fn main() -> Result<(), io::Error> {
    // like. asyncio.run(coro()) in python
    tokio::runtime::Runtime::new()?.block_on(async_func());

    let ret = async_main_macro();
    println!("result : {}", ret);

    Ok(())
}

async fn gather_example() {
    // like. asyncio.gather(coro1(), coro2()) in python
    let start = Instant::now();
    let ret = tokio::join!(sleep_msec(1000), sleep_msec(500), sleep_msec(700),);
    let elapsed_time = start.elapsed().as_millis();
    println!(">>> {:?}. elapsed_time {:?} [msec]", ret, &elapsed_time);
}

async fn create_task_example() {
    // like. create_task
    let start = Instant::now();

    let task = tokio::spawn(sleep_msec(100));
    sleep_msec(200).await;

    let ret = task.await.unwrap();
    println!(
        "{:?}. elapsed time = {} [msec] ",
        ret,
        start.elapsed().as_millis()
    );
}

async fn run_in_executor_example() {
    let start = Instant::now();
    let blocking_task = tokio::task::spawn_blocking(|| {
        use std::thread;
        use std::time::Duration;

        thread::sleep(Duration::from_millis(100));
    });

    blocking_task.await.unwrap();
    println!(
        "run_in_executor. elapsed time = {} [msec] ",
        start.elapsed().as_millis()
    );
}

async fn async_func() {
    let greeting: String = hello().await;
    println!("{}", greeting);

    // like. "await asyncio.wait_for(sleep_msec(), timeout=0.2)" in python
    let res = tokio::time::timeout(time::Duration::from_millis(200), sleep_msec(1000)).await;
    assert!(res.is_err());

    gather_example().await;
    create_task_example().await;
    run_in_executor_example().await;
}

#[tokio::main]
async fn async_main_macro() -> u32 {
    sleep_msec(1000).await;

    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn async_test() {
        assert_eq!(sleep_msec(100).await, 100);
    }
}
