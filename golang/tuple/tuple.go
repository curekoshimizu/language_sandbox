package tuple

type Tuple[T1, T2 any] struct {
    first T1
    second T2
}

func New[T1, T2 any](first T1, second T2) Tuple[T1, T2] {
    return Tuple[T1, T2]{
        first: first,
        second: second,
    }
}

func (t Tuple[T1, T2]) First() T1 {
    return t.first
}

func (t Tuple[T1, T2]) Second() T2 {
    return t.second
}
