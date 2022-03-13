package defer_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDeferOrder(t *testing.T) {
	queue := make(chan int, 4)

	func() {
		defer func() {
			queue <- 3
			queue <- 4
		}()
		defer func() {
			queue <- 1
			queue <- 2
		}()

	}()

	assert.Equal(t, 1, <-queue)
	assert.Equal(t, 2, <-queue)
	assert.Equal(t, 3, <-queue)
	assert.Equal(t, 4, <-queue)
}
