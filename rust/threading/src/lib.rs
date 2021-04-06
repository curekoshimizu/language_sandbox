#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::RwLock;
    use std::thread;
    use std::time::Duration;
    use std::time::Instant;

    const NUM_THREADS: usize = 20000;

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
        // Rc/(RefCell or Cell) <--> Arc/Mutex

        {
            // RefCell Example.
            use std::cell::RefCell;
            let counter = RefCell::new(0);
            {
                let mut a = counter.borrow_mut();
                *a += 1;
                // counter.borrow_mut(); // already borrowed: BorrowMutError occurs!!!!
                // In that case, we need Reference Counter.
            }
            assert_eq!(*counter.borrow(), 1);
        }
        {
            // Use Cell instead of RefCell.
            // Cell<T> requires T to implement "Copy".
            // The method get returns a copy of the contained value,
            // and set stores a copy of the argument val as the new value.
            // Meanwhile, it won't throw runtime error in comparison with RefCell.
            use std::cell::Cell;
            let counter = Cell::new(0);
            {
                let mut a = counter.get();
                a += 1;
                counter.set(a);
                let mut b = counter.get();
                b += 1;
                counter.set(b);
            }
            assert_eq!(counter.get(), 2);
        }

        {
            // Use Rc and RefCell
            use std::cell::RefCell;
            use std::rc::Rc;
            let counter = Rc::new(RefCell::new(0));

            let mut f_vec = vec![];

            for _ in 0..10 {
                let counter = Rc::clone(&counter);
                let f = move || {
                    let mut num = counter.borrow_mut();

                    *num += 1;
                };
                f_vec.push(f);
            }
            assert_eq!(Rc::strong_count(&counter), 11);

            for f in f_vec {
                f();
            }

            assert_eq!(*counter.borrow(), 10);
        }
        {
            // Use Arc and Mutex
            // NOTE: Arc
            //  Atomic Reference Counter.
            //  Rc does not implement "Send" and "Sync"
            let start = Instant::now();
            let counter = Arc::new(Mutex::new(0));
            let mut handles = vec![];

            for _ in 0..NUM_THREADS {
                // NOTE:
                //  These codes are same. But Arc::clone is better because x.clone() makes us think
                //  depp copy which makes programs slow.
                // let counter = counter.clone();
                let counter = Arc::clone(&counter);
                let handle = thread::spawn(move || {
                    let mut num = counter.lock().unwrap();

                    *num += 1;
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            assert_eq!(*counter.lock().unwrap(), NUM_THREADS);
            println!("Arc+Mutex : {} msec", start.elapsed().as_millis())
        }
        {
            let start = Instant::now();

            // Use Arc and RwLock
            let counter = Arc::new(RwLock::new(0));
            let mut handles = vec![];

            for _ in 0..NUM_THREADS {
                let counter = Arc::clone(&counter);
                let handle = thread::spawn(move || {
                    let mut num = counter.write().unwrap();

                    *num += 1;
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            assert_eq!(*counter.read().unwrap(), NUM_THREADS);
            println!("Arc+RwLock : {} msec", start.elapsed().as_millis())
        }
        {
            // Use Atomic
            let start = Instant::now();
            let counter = Arc::new(AtomicUsize::new(0));
            let mut handles = vec![];

            for _ in 0..NUM_THREADS {
                let counter = Arc::clone(&counter);
                let handle = thread::spawn(move || {
                    // NOTE: what is Ordering?
                    // ref.
                    //  * https://blog.tiqwab.com/2020/05/13/memory-consistency.html
                    //  * C++ pocket reference
                    counter.fetch_add(1, Ordering::SeqCst);
                });
                handles.push(handle);
            }

            for handle in handles {
                handle.join().unwrap();
            }

            assert_eq!(counter.load(Ordering::SeqCst), NUM_THREADS);
            println!("Arc+AtomicUsize : {} msec", start.elapsed().as_millis())
        }
    }
}
