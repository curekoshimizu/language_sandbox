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
// asyncio.generator --> tokio_streams?
// web server

fn main() -> Result<(), io::Error> {
    // like. asyncio.run(coro()) in python
    // but. not recommend this way,
    // because actual tokio::main macro is more complicated.
    // please use cargo expand to check the macro.
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
    use std::thread;

    #[tokio::test]
    async fn async_test() {
        assert_eq!(sleep_msec(100).await, 100);
    }

    async fn non_blocking(sleep_time: u64) {
        let start = Instant::now();
        time::sleep(time::Duration::from_millis(sleep_time)).await;
        println!(
            "[id={:?}] async sleep elapsed time = {} msec. expect = {}",
            thread::current().id(),
            start.elapsed().as_millis(),
            sleep_time,
        );
    }
    async fn blocking() {
        use std::time::Duration;

        let start = Instant::now();
        let expect = 300;
        thread::sleep(Duration::from_millis(expect));
        println!(
            "[id={:?}] blocking sleep elapsed time = {} msec. expect = {}",
            thread::current().id(),
            start.elapsed().as_millis(),
            expect,
        );
    }

    #[test]
    fn async_test_with_blocking() {
        #[tokio::main]
        async fn async_main() {
            let start = Instant::now();

            tokio::join!(non_blocking(100), non_blocking(50), blocking(),);

            println!("{} msec", start.elapsed().as_millis());
        }

        async_main();
    }

    #[test]
    fn async_test_with_blocking_2() {
        #[tokio::main]
        async fn async_main() {
            let start = Instant::now();

            let task_1 = tokio::spawn(non_blocking(100));
            let task_2 = tokio::spawn(non_blocking(50));
            let task_3 = tokio::spawn(blocking());

            if let (Ok(_), Ok(_), Ok(_)) = tokio::join![task_1, task_2, task_3] {
                println!("{} msec", start.elapsed().as_millis());
            } else {
                panic!("join failed...");
            }
        }

        async_main();
    }

    #[tokio::test]
    async fn async_mutex() {
        use threading::NUM_THREADS;

        const LOOP: usize = NUM_THREADS;

        use std::sync::Arc;
        use tokio::sync::Mutex;
        use tokio::sync::RwLock;

        {
            let start = Instant::now();
            let counter = Arc::new(Mutex::new(0));
            let mut handles = vec![];

            for _ in 0..LOOP {
                let counter = Arc::clone(&counter);
                let handle = tokio::spawn(async move {
                    let mut num = counter.lock().await;

                    *num += 1;
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }

            assert_eq!(*counter.lock().await, LOOP);
            println!("Async Arc+Mutex : {} msec", start.elapsed().as_millis())
        }
        {
            let start = Instant::now();
            let counter = Arc::new(Mutex::new(0));
            let mut handles = vec![];

            for _ in 0..LOOP {
                let counter = Arc::clone(&counter);
                let handle = tokio::spawn(async move {
                    let mut num = counter.lock().await;

                    *num += 1;
                });
                handles.push(handle);
            }

            futures::future::join_all(handles).await;

            assert_eq!(*counter.lock().await, LOOP);
            println!(
                "Async Arc+Mutex (join_all) : {} msec",
                start.elapsed().as_millis()
            )
        }
        {
            let start = Instant::now();

            // Use Arc and RwLock
            let counter = Arc::new(RwLock::new(0));
            let mut handles = vec![];

            for _ in 0..LOOP {
                let counter = Arc::clone(&counter);
                let handle = tokio::spawn(async move {
                    let mut num = counter.write().await;

                    *num += 1;
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.await.unwrap();
            }

            assert_eq!(*counter.read().await, LOOP);
            println!("Async Arc+RwLock : {} msec", start.elapsed().as_millis())
        }
        {
            // Use futures::future::join_all

            let start = Instant::now();
            let counter = Arc::new(RwLock::new(0));
            let mut handles = vec![];

            for _ in 0..LOOP {
                let counter = Arc::clone(&counter);
                let handle = tokio::spawn(async move {
                    let mut num = counter.write().await;

                    *num += 1;
                });
                handles.push(handle);
            }

            futures::future::join_all(handles).await;

            assert_eq!(*counter.read().await, LOOP);
            println!(
                "Async Arc+RwLock (join_all) : {} msec",
                start.elapsed().as_millis()
            )
        }
    }
}
