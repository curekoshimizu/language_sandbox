package async_await

import (
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
)


func BlockSleep(msec int) int {
	time.Sleep( time.Duration(msec) * time.Millisecond)
	return msec
}

func TestAsyncAwait(t *testing.T) {
    now:=time.Now()


    msec := 100
	future := Exec(func() interface{} {
		return BlockSleep(msec)
	})
	val := future.Await().(int)
    assert.Equal(t, val, msec)

    ans := time.Since(now).Milliseconds()
    expect := int64(msec) *90/100 // 90%
    assert.True(t, ans > expect, "expect :  %v > %v", ans, expect)

}
