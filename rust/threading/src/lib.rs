#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn simple_thread_test() {
        // use lambda
        {
            let func = || {
                for i in 1..=10 {
                    println!("hi number {} from the spawned thread!", i);
                    thread::sleep(Duration::from_millis(1));
                }
            };

            // another thread runs!
            let handle = thread::spawn(func);
            handle.join().unwrap();
        }

        fn hello_func() -> String {
            "hello".to_owned()
        }

        // use fn
        {
            let handle = thread::spawn(hello_func);
            let result = handle.join().unwrap();
            assert_eq!(result, "hello");
        }
    }

    #[test]
    fn channel_test() {
        // single message
        {
            let (tx, rx) = mpsc::channel();

            // move keyword is needed because tx variable is used from spawned thread.
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(10));
                let val = String::from("hello");
                tx.send(val).unwrap();
            });
            // waiting until the message from the spawned thread is received.
            let received = rx.recv().unwrap();
            assert_eq!(received, "hello");
        }

        // multiple message
        {
            let (tx1, rx) = mpsc::channel();
            let tx2 = tx1.clone();

            // tx1 is used for thread1
            // tx2 is used for thread2

            thread::spawn(move || {
                // thread1
                let vals = vec![
                    String::from("hi"),
                    String::from("from"),
                    String::from("the"),
                    String::from("thread1"),
                ];

                for val in vals {
                    tx1.send(val).unwrap();
                    thread::sleep(Duration::from_millis(10));
                }
            });

            thread::spawn(move || {
                // thread2
                let vals = vec![
                    String::from("hi"),
                    String::from("from"),
                    String::from("the"),
                    String::from("thread2"),
                ];

                for val in vals {
                    tx2.send(val).unwrap();
                    thread::sleep(Duration::from_millis(5));
                }
            });

            let mut msg = String::new();
            for received in rx {
                println!("Got: {}", received);
                msg.push_str(&received);
            }
            assert!(msg.contains("thread1"));
            assert!(msg.contains("thread2"));
        }
    }

    #[test]
    fn simple_mutex() {
        // use only the main thread

        let m: Mutex<Vec<u32>> = Mutex::new(vec![1, 2, 3]);
        {
            let mut num = m.lock().unwrap();
            *num.get_mut(0).unwrap() = 6;
        }
        assert_eq!(*m.lock().unwrap(), vec![6, 2, 3]);
    }

    #[test]
    fn mutex_thread() {
        // NOTE: Arc
        //  Atomic Reference Counter.
        //  Rc does not implement "Send" and "Sync"
        let counter = Arc::new(Mutex::new(0));
        let mut handles = vec![];

        for _ in 0..10 {
            let counter = counter.clone();
            let handle = thread::spawn(move || {
                let mut num = counter.lock().unwrap();

                *num += 1;
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(*counter.lock().unwrap(), 10);
    }
}
