package tuple

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTuple(t *testing.T) {
	tuple := New(1, "a")
	assert.Equal(t, tuple.First(), 1)
	assert.Equal(t, tuple.Second(), "a")
}
