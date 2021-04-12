#[cfg(test)]
mod tests {
    #[test]
    fn iter_product() {
        let a = 0..5;
        let b = 0..3;

        let product: Vec<(i32, i32)> = a //
            .flat_map(|x| b.clone().map(move |y| (x, y)))
            .collect();
        assert_eq!(
            product,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
                (3, 0),
                (3, 1),
                (3, 2),
                (4, 0),
                (4, 1),
                (4, 2),
            ]
        );
    }

    #[test]
    fn iter_product_three() {
        let a = 0..4;
        let b = 0..3;
        let c = 0..2;

        let product: Vec<(i32, i32, i32)> = a
            .flat_map(|x| {
                b.clone().flat_map({
                    let c = &c;
                    move |y| c.clone().map(move |z| (x, y, z))
                })
            })
            .collect();

        assert_eq!(
            product,
            vec![
                (0, 0, 0),
                (0, 0, 1,),
                (0, 1, 0),
                (0, 1, 1,),
                (0, 2, 0),
                (0, 2, 1,),
                (1, 0, 0),
                (1, 0, 1,),
                (1, 1, 0),
                (1, 1, 1,),
                (1, 2, 0),
                (1, 2, 1,),
                (2, 0, 0),
                (2, 0, 1,),
                (2, 1, 0),
                (2, 1, 1,),
                (2, 2, 0),
                (2, 2, 1,),
                (3, 0, 0),
                (3, 0, 1,),
                (3, 1, 0),
                (3, 1, 1,),
                (3, 2, 0),
                (3, 2, 1,),
            ]
        );
    }
}
