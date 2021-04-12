#[cfg(test)]
mod tests {
    #[test]
    fn capacity_resize() {
        let mut vec: Vec<usize> = Vec::with_capacity(5);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 5);
        assert_eq!(vec, []);

        vec.resize(5, 0);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.capacity(), 5);
        assert_eq!(vec, [0, 0, 0, 0, 0]);
    }

    fn capacity_push() {
        let mut vec: Vec<usize> = Vec::with_capacity(5);
        assert_eq!(vec.len(), 0);
        assert_eq!(vec.capacity(), 5);
        assert_eq!(vec, []);

        vec.push(0);
        vec.push(1);
        vec.push(2);
        vec.push(3);
        vec.push(4);
        assert_eq!(vec.len(), 5);
        assert_eq!(vec.capacity(), 5);
        assert_eq!(vec, (0..5).collect::<Vec<usize>>());
    }
}
