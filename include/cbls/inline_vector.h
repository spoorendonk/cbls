#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace cbls {

/// A vector that holds its first `N` elements inline and moves to the heap only
/// past them (#274). For `Move::changes`: a generator builds one `Move` per
/// candidate it proposes, almost every one has one or two changes, and a
/// `std::vector` there cost a malloc/free pair per candidate -- 6-10% of CVRP's
/// instructions, its only allocator traffic in the hot loop.
///
/// Two storage modes, never mixed: `size() <= N` lives in `inline_`, and
/// `size() > N` lives entirely in `heap_`, with every inline slot back at `T{}`.
/// So the elements are always contiguous, `data()` is one branch, and nothing
/// is placement-constructed.
///
/// INVARIANT: an inline slot at or past `size()` holds `T{}`. That is what makes
/// the defaulted copy cheap -- copying an empty slot of a `Move::Change` copies
/// an empty `replacement` -- and it is why `clear()` resets the slots it used
/// rather than only zeroing the count.
///
/// Cost: `N * sizeof(T)` bytes inline whatever the size, and a move costs
/// `N` element moves where a `std::vector` steals one pointer. That loses only
/// where elements are many and moved often; a `Move` has at most two changes on
/// every path a generator builds, and the candidate vector it is moved into is
/// reused across batches, so the trade is one ~160-byte copy for one malloc and
/// one free.
template <typename T, size_t N>
class InlineVector {
    static_assert(N > 0, "an InlineVector with no inline slots is a std::vector");

public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    InlineVector() = default;
    InlineVector(std::initializer_list<T> init) {
        for (const T& v : init) {
            push_back(v);
        }
    }
    ~InlineVector() = default;
    InlineVector(const InlineVector&) = default;
    InlineVector& operator=(const InlineVector&) = default;
    /// A moved-from InlineVector is EMPTY, as a moved-from std::vector is in
    /// practice: the defaulted move would keep `size_` over moved-from slots and,
    /// past `N`, over an emptied `heap_`. noexcept, so a `std::vector<Move>`
    /// moves its elements when it grows rather than copying them.
    InlineVector(InlineVector&& other) noexcept
        : inline_(std::move(other.inline_)), heap_(std::move(other.heap_)), size_(other.size_) {
        other.reset_moved_from();
    }
    InlineVector& operator=(InlineVector&& other) noexcept {
        if (this != &other) {
            inline_ = std::move(other.inline_);
            heap_ = std::move(other.heap_);
            size_ = other.size_;
            other.reset_moved_from();
        }
        return *this;
    }

    /// Copies first: `v` may be one of this vector's own elements, which the
    /// spill at size `N` moves away.
    void push_back(const T& v) { push_back(T(v)); }
    /// `v` must not be one of this vector's own elements.
    void push_back(T&& v) { emplace_slot() = std::move(v); }

    void clear() {
        if (on_heap()) {
            heap_.clear();
        } else {
            for (size_t i = 0; i < size_; ++i) {
                inline_[i] = T{};
            }
        }
        size_ = 0;
    }

    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] bool empty() const { return size_ == 0; }

    [[nodiscard]] T* data() { return on_heap() ? heap_.data() : inline_.data(); }
    [[nodiscard]] const T* data() const { return on_heap() ? heap_.data() : inline_.data(); }

    iterator begin() { return data(); }
    iterator end() { return data() + size_; }
    [[nodiscard]] const_iterator begin() const { return data(); }
    [[nodiscard]] const_iterator end() const { return data() + size_; }

    T& operator[](size_t i) { return data()[i]; }
    const T& operator[](size_t i) const { return data()[i]; }
    T& front() { return data()[0]; }
    [[nodiscard]] const T& front() const { return data()[0]; }
    T& back() { return data()[size_ - 1]; }
    [[nodiscard]] const T& back() const { return data()[size_ - 1]; }

private:
    [[nodiscard]] bool on_heap() const { return size_ > N; }

    void reset_moved_from() noexcept {
        // Default construction is not asserted: for a nested `T` with default
        // member initialisers (`Move::Change`) clang evaluates the trait before
        // the enclosing class is complete and caches `false`. `T{}` throwing
        // here would terminate.
        static_assert(std::is_nothrow_move_assignable_v<T>,
                      "the noexcept move resets moved-from slots to T{}");
        for (T& slot : inline_) {
            slot = T{};
        }
        heap_.clear();
        size_ = 0;
    }

    /// The slot the next element goes in, `size_` already counting it.
    T& emplace_slot() {
        if (size_ < N) {
            return inline_[size_++];
        }
        if (size_ == N) {
            // Spill: the inline elements move to the heap, in order, and their
            // slots go back to T{} to keep the invariant.
            heap_.reserve(2 * N);
            for (T& slot : inline_) {
                heap_.push_back(std::move(slot));
                slot = T{};
            }
        }
        heap_.emplace_back();
        ++size_;
        return heap_.back();
    }

    std::array<T, N> inline_{};
    std::vector<T> heap_;
    size_t size_ = 0;
};

}  // namespace cbls
