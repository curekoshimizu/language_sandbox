package async_await

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func BlockSleep(msec int) int {
	time.Sleep(time.Duration(msec) * time.Millisecond)
	return msec
}

func TestAsyncAwait(t *testing.T) {
	now := time.Now()

	msec := 100
	future := Async(func() interface{} {
		return BlockSleep(msec)
	})
	defer future.Close()
	val := future.Await(context.Background()).(int)
	assert.Equal(t, val, msec)

	ans := time.Since(now).Milliseconds()
	expect := int64(msec) * 90 / 100 // 90%
	assert.True(t, ans > expect, "expect :  %v > %v", ans, expect)
}

func TestTimeoutAsyncAwait(t *testing.T) {
	now := time.Now()

	sleepTime := 500
	timeout := 100
	future := Async(func() interface{} {
		return BlockSleep(sleepTime)
	})
	defer future.Close()

	context, cancel := context.WithTimeout(
		context.Background(), time.Duration(timeout)*time.Millisecond)
	defer cancel()

	_, ok := future.Await(context).(int)
	assert.Equal(t, ok, false)
	_, ok = future.Await(context).(error)
	assert.Equal(t, ok, true)

	ans := time.Since(now).Milliseconds()
	expect := int64(timeout) * 90 / 100 // 90%
	assert.True(t, ans > expect, "expect :  %v > %v", ans, expect)
}

func TestClose(t *testing.T) {
	sleepTime := 500
	timeout := 100
	future := Async(func() interface{} {
		return BlockSleep(sleepTime)
	})
	defer func() {
		now := time.Now()
		future.Close()

		ans := time.Since(now).Milliseconds()
		expect := int64(timeout) * 90 / 100 // 90%
		assert.True(t, ans > expect, "expect :  %v > %v", ans, expect)

		val, ok := future.Await(context.Background()).(int)
		assert.Equal(t, ok, true)
		assert.Equal(t, val, sleepTime)
	}()
	BlockSleep(timeout)
}
