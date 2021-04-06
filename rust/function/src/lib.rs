// NOTE: Roughly speaking.
//   fn : function pointer type
//   Fn/FnMut/FnOnce : Closure type (trait objects)
//      Both FnMut and FnOnce are supertraits of Fn.

pub fn squared(x: i32) -> i32 {
    x * x
}

pub fn cubed(x: i32) -> i32 {
    x * x * x
}

pub fn returns_closure() -> Box<dyn Fn(i32) -> i32> {
    Box::new(|x| x + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer() {
        use std::ops::Deref;

        fn as_raw_bytes<'a, T: ?Sized>(x: &'a T, n: usize) -> &'a [u8] {
            assert_eq!(std::mem::size_of_val(x), n);

            unsafe {
                std::slice::from_raw_parts(x as *const T as *const u8, std::mem::size_of_val(x))
            }
        }

        pub struct S<T: ?Sized> {
            pub x: u8,
            pub y: T,
        }

        let array: [u16; 5] = [1, 2, 3, 4, 5];
        let array_ptr = &array as *const [u16; 5];
        println!("array_ref    = {:?}", as_raw_bytes(&array, 10)); // [u16; 5]
        println!("&array_ref   = {:?}", as_raw_bytes(&&array, 8)); //&[u16; 5]
        println!("&arrary_ptr  = {:?}", as_raw_bytes(&array_ptr, 8)); //*const [u16; 5]

        println!("[slice] (non sized object)");
        let array_slice = &array[..];
        let array_slice_ptr = &array[..] as *const [u16];

        println!("array_slice  = {:?}", as_raw_bytes(array_slice, 10)); // [u16]
        println!("&array_slice = {:?}", as_raw_bytes(&array_slice, 16)); //&[u16] (fat pointer)
        println!(
            "&array_slice_ptr = {:?}",
            as_raw_bytes(&array_slice_ptr, 16)
        ); //*const [u16] (fat pointer)

        println!("[string slice]");
        let str_slice = "HelloWorld";
        println!("str_slice    = {:?}", as_raw_bytes(str_slice, 10)); // &str
        println!("&str_slice   = {:?}", as_raw_bytes(&str_slice, 16)); //&&str (fat pointer)

        println!("[closure (trait object)]");
        let closure = |x, y| x + y;
        let closure_ref: &dyn Fn(u8, u8) -> u8 = &closure;

        let box_closure: Box<dyn Fn(u8, u8) -> u8> = Box::new(closure);

        println!("closure_ref  = {:?}", as_raw_bytes(closure_ref, 0)); // closure
        println!("&closure_ref = {:?}", as_raw_bytes(&closure_ref, 16)); //&closure (fat pointer)
        println!("box_closure_ref  = {:?}", as_raw_bytes(&box_closure, 16)); // box closure

        println!("[mutable closure (trait object)]");
        let mut a = 0;
        let mut_closure = |x| {
            a += x;
            a
        };
        let mut_closure_ref: &dyn FnMut(u8) -> u8 = &mut_closure;
        println!(
            "mut_closure_ref     = {:?}",
            as_raw_bytes(mut_closure_ref, 8)
        ); // mutable_closure
        println!(
            "&mut_closure_ref    = {:?}",
            as_raw_bytes(&mut_closure_ref, 16)
        ); //&mutable_closure (fat pointer)

        println!("[structure which has an unsized object]");
        let sarr: Box<S<[u16; 3]>> = Box::new(S { x: 3, y: [1, 2, 3] });
        let sarray_ref = sarr.deref();
        let sslice: Box<S<[u16]>> = Box::new(S { x: 3, y: [1, 2, 3] });
        let sslice_ref = sslice.deref();
        println!("sarray_ref    = {:?}", as_raw_bytes(sarray_ref, 8)); //&S<[u16; _]>
        println!("&sarray_ref   = {:?}", as_raw_bytes(&sarray_ref, 8)); //&&S<[u16; _]>
        println!("sslice_ref    = {:?}", as_raw_bytes(sslice_ref, 8)); //&S<[u16]>
        println!("&sslice_ref   = {:?}", as_raw_bytes(&sslice_ref, 16)); //&&S<[u16]> (fat pointer)
    }

    #[test]
    fn fn_pointer() {
        // The size of f is 0!
        // f actual type is `fn(_) -> _ {squared}`, which includes function infomation.
        // That's why its size is 0 byte!
        {
            let f = squared;
            assert_eq!(f(2), 4);

            assert_eq!(std::mem::size_of_val(&f), 0);
        }

        // The size of f is 8, because it is a pure function pointer.
        // same as the size of address is 64bit (8Bytes) on x86_64.
        {
            let mut f: fn(i32) -> i32 = squared;
            assert_eq!(f(2), 4);
            f = cubed;
            assert_eq!(f(2), 8);

            assert_eq!(std::mem::size_of_val(&f), 8);
        }

        // The size of f is 16, because it includes fat pointer which includes fat pointer.
        {
            let mut f: &dyn Fn(i32) -> i32 = &squared;
            assert_eq!(f(2), 4);
            f = &cubed;
            assert_eq!(f(2), 8);

            assert_eq!(std::mem::size_of_val(&f), 16);
        }

        {
            let mut f: Box<dyn Fn(i32) -> i32> = Box::new(squared);
            assert_eq!(f(2), 4);
            f = Box::new(cubed);
            assert_eq!(f(2), 8);

            assert_eq!(std::mem::size_of_val(&f), 16);
        }
    }

    #[test]
    fn fn_pointer_function() {
        fn multi_applied(f: fn(i32) -> i32, x: i32) -> i32 {
            f(f(x))
        }

        assert_eq!(squared(3), 9);
        assert_eq!(multi_applied(squared, 3), 81);

        // non-captured closure can be treated as a function pointer type
        let add_one = |x| x + 1;
        assert_eq!(add_one(3), 4);
        assert_eq!(multi_applied(add_one, 3), 5);

        let y = 3;
        let add = move |x| x + y;
        assert_eq!(add(3), 6);

        let id = &|x| x;
        assert_eq!(id(3), 3);
        // NOTE: compile error
        //   Because closure reference is not fn pointer.
        //   As you know, in its case, we need a far pointer.
        // assert_eq!(multi_applied(id, 3), 3);

        fn _returns_closure() -> Box<dyn Fn(i32) -> i32> {
            Box::new(|x| x + 1)
        }
        // NOTE: compile error
        //   Because Box::new(|x| x + 1) is not fn pointer.
        //   As you know, in its case, we need a far pointer.
        // assert_eq!(multi_applied(_returns_closure(), 3), 5);
    }

    #[test]
    fn closure_trait() {
        // NOTE:
        //
        // This code is wrong. Because "Fn" is a trait object.
        // fn multi_applied(f: Fn(i32) -> i32, x: i32) -> i32 {
        //     f(f(x))
        // }

        fn multi_applied<F: Fn(i32) -> i32>(f: F, x: i32) -> i32 {
            f(f(x))
        }

        assert_eq!(squared(3), 9);
        assert_eq!(multi_applied(squared, 3), 81);

        // non-captured closure can be treated as a function pointer type
        let add_one = |x| x + 1;
        assert_eq!(add_one(3), 4);
        assert_eq!(multi_applied(add_one, 3), 5);

        let y = 3;
        let add = move |x| x + y;
        assert_eq!(add(3), 6);

        let id = &|x| x;
        assert_eq!(id(3), 3);
        assert_eq!(multi_applied(id, 3), 3);

        let f = returns_closure();
        assert_eq!(multi_applied(f, 3), 5);
    }
}
