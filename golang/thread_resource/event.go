package thread_resource

import (
	"context"
	"sync"
)

type Event struct {
	value bool
	ch    chan struct{}
	lock  sync.RWMutex
}

func NewEvent() Event {
	return Event{value: false, ch: make(chan struct{})}
}

func (e *Event) IsSet() bool {
	e.lock.RLock()
	defer e.lock.RUnlock()

	return e.value
}

func (e *Event) Set() {
	e.lock.Lock()
	defer e.lock.Unlock()

	if !e.value {
		e.value = true
		close(e.ch)
	}
}

func (e *Event) Clear() {
	e.lock.Lock()
	defer e.lock.Unlock()

	if e.value {
		// just assert code: e.ch must be closed.
		<-e.ch

		e.value = false
		e.ch = make(chan struct{})
	}
}

func (e *Event) Wait(ctx context.Context) bool {
	select {
	case <-ctx.Done():
		return false
	case <-e.wait():
		return true
	}
}

func (e *Event) wait() chan struct{} {
	e.lock.RLock()
	defer e.lock.RUnlock()
	return e.ch
}
