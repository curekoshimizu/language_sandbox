package async_await

import (
	"context"
)

type Future[T any] struct {
	Await func(ctx context.Context) (T, error)
	Close func()
}

func Async[T any](f func() T) Future[T] {
	var result T
	hasFutureFinished := make(chan struct{})
	go func() {
		defer close(hasFutureFinished)
		result = f() // context を渡せるようにすべき
	}()
	future := Future[T]{
		Await: func(ctx context.Context) (T, error) {
			select {
			case <-hasFutureFinished:
				return result, nil
			case <-ctx.Done():
				return result, ctx.Err()
			}
		},
		Close: func() {
			<-hasFutureFinished
		},
	}

	return future
}
