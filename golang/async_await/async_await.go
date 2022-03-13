package async_await

import (
	"context"
)

type Future struct {
	Await func(ctx context.Context) interface{}
	Close func()
}

func Async(f func() interface{}) Future {
	var result interface{}
	hasFutureFinished := make(chan struct{})
	go func() {
		defer close(hasFutureFinished)
		result = f() // context を渡せるようにすべき
	}()
	future := Future{
		Await: func(ctx context.Context) interface{} {
			select {
			case <-hasFutureFinished:
				return result
			case <-ctx.Done():
				return ctx.Err()
			}
		},
		Close: func() {
			<-hasFutureFinished
		},
	}

	return future
}
