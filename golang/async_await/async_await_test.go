package async_await

import (
	"context"
	"testing"
	"time"

	"github.com/curekoshimizu/language_sandbox/golang/thread_resource"
	"github.com/stretchr/testify/assert"
)

func BlockSleep(msec int) int {
	time.Sleep(time.Duration(msec) * time.Millisecond)
	return msec
}

func TestAsyncAwait(t *testing.T) {
	event := thread_resource.NewEvent()
	future := Async(func() interface{} {
		event.Wait(context.Background())
		return true
	})
	defer future.Close()

	go func() {
		BlockSleep(100)
		event.Set()
	}()

	ret := future.Await(context.Background()).(bool)
	assert.True(t, ret)
	assert.True(t, event.IsSet())
}

func TestTimeoutAsyncAwait(t *testing.T) {
	event := thread_resource.NewEvent()
	timeout := 100
	future := Async(func() interface{} {
		event.Wait(context.Background())
		return true
	})
	defer future.Close()

	context, cancel := context.WithTimeout(
		context.Background(), time.Duration(timeout)*time.Millisecond)
	defer cancel()

	_, ok := future.Await(context).(bool)
	assert.Equal(t, ok, false)
	_, ok = future.Await(context).(error)
	assert.Equal(t, ok, true)

	event.Set()
}

func TestClose(t *testing.T) {
	event := thread_resource.NewEvent()
	future := Async(func() interface{} {
		BlockSleep(100)
		event.Set()
		return true
	})

	assert.False(t, event.IsSet())
	future.Close()
	assert.True(t, event.IsSet())

	_, ok := future.Await(context.Background()).(bool)
	assert.Equal(t, ok, true)
}
