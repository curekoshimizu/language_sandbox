package async_await

import (
	"context"
	"testing"
	"time"

	"github.com/barweiss/go-tuple"
	"github.com/curekoshimizu/language_sandbox/golang/thread_resource"
	"github.com/stretchr/testify/assert"
)

func BlockSleep(msec int) int {
	time.Sleep(time.Duration(msec) * time.Millisecond)
	return msec
}

func TestAsyncAwait(t *testing.T) {
	event := thread_resource.NewEvent()
	future := Async(func() bool {
		event.Wait(context.Background())
		return true
	})
	defer future.Close()

	go func() {
		BlockSleep(100)
		event.Set()
	}()

	ret, err := future.Await(context.Background())
	assert.True(t, ret)
	assert.Nil(t, err)
	assert.True(t, event.IsSet())
}

func TestTimeoutAsyncAwait(t *testing.T) {
	event := thread_resource.NewEvent()
	timeout := 100
	future := Async(func() bool {
		event.Wait(context.Background())
		return true
	})
	defer future.Close()

	context, cancel := context.WithTimeout(
		context.Background(), time.Duration(timeout)*time.Millisecond)
	defer cancel()

	_, err := future.Await(context)
	assert.NotNil(t, err)

	event.Set()
}

func TestClose(t *testing.T) {
	event := thread_resource.NewEvent()
	future := Async(func() bool {
		BlockSleep(100)
		event.Set()
		return true
	})

	assert.False(t, event.IsSet())
	future.Close()
	assert.True(t, event.IsSet())

	_, err := future.Await(context.Background())
	assert.Nil(t, err)
}

func TestTuple(t *testing.T) {
	f := func() (int, string) {
		return 1, "a"
	}
	future := Async(func() tuple.T2[int, string]{
		return tuple.New2(f())
	})
    defer future.Close()
    val, err := future.Await(context.Background())
    if err != nil {
        intVal, strVal := val.Values()
        assert.Equal(t, intVal, 1)
        assert.Equal(t, strVal, "a")
    }
}
