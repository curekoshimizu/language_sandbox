package thread_resource

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestEvent(t *testing.T) {
	event := NewEvent()
	assert.False(t, event.IsSet())
	event.Set()
	assert.True(t, event.IsSet())
	event.Clear()
	assert.False(t, event.IsSet())

	go func() {
		time.Sleep(time.Duration(100) * time.Millisecond)
		event.Set()
	}()
	ret := event.Wait(context.Background())
	assert.True(t, ret)
	assert.True(t, event.IsSet())
	event.Clear()
	assert.False(t, event.IsSet())

	// timeout will occur
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(100)*time.Millisecond)
	defer cancel()
	ret = event.Wait(ctx)
	assert.False(t, ret)
	assert.False(t, event.IsSet())

	event = NewEvent()
	event.Clear()
	event.Clear()
	event.Clear()
	event.Clear()
	assert.False(t, event.IsSet())
	event.Set()
	event.Set()
	event.Set()
	event.Set()
	assert.True(t, event.IsSet())
	event.Clear()
	event.Clear()
	event.Clear()
	event.Clear()
	assert.False(t, event.IsSet())
}
