# 1. https://leetcode-cn.com/problems/number-of-1-bits/
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n != 0:
            n &= (n - 1)
            count += 1
        return count

# 2. https://leetcode-cn.com/problems/power-of-two/
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        return n & (n - 1) == 0

# 3. https://leetcode-cn.com/problems/reverse-bits/
class Solution:
    def reverseBits(self, n: int) -> int:
        res = n & 1
        for _ in range(31):
            n = n >> 1
            res = res << 1
            res |= (n & 1)
        return res

# 4. https://leetcode-cn.com/problems/relative-sort-array/
class Solution:
    def relativeSortArray(self, arr1: List[int], arr2: List[int]) -> List[int]:
        count = {}
        for i in arr1:
            if i not in count:
                count[i] = 1
            else:
                count[i] += 1
        p = 0
        for i in arr2:
            if i in count:
                for _ in range(count[i]):
                    arr1[p] = i
                    p += 1
                count.pop(i)
            else:
                break
        other = sorted(list(count.keys()))
        for i in other:
            for _ in range(count[i]):
                arr1[p] = i
                p += 1
            count.pop(i)
        return arr1

# 5. https://leetcode-cn.com/problems/valid-anagram/
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        count = {}
        for c in s:
            if c in count:
                count[c] += 1
            else:
                count[c] = 1
        for c in t:
            if c in count:
                count[c] -= 1
            else:
                return False
        for _, v in count.items():
            if v != 0:
                return False
        return True

# 6. https://leetcode-cn.com/problems/lru-cache/#/
class DoubleLinkedNode:
    def __init__(self, key=None, val=None):
        self.key = key
        self.val = val
        self.pre = None
        self.nxt = None


class LRUCache:
    def __init__(self, capacity: int):
        self.head = DoubleLinkedNode()
        self.tail = DoubleLinkedNode()
        self.head.nxt = self.tail
        self.tail.pre = self.head
        self.capacity = capacity
        self.key_pos = {}

    def get(self, key: int) -> int:
        if key not in self.key_pos:
            return -1
        node = self.key_pos[key]
        node.pre.nxt = node.nxt
        node.nxt.pre = node.pre
        first = self.head.nxt
        self.head.nxt = node
        node.pre = self.head
        node.nxt = first
        first.pre = node
        self.key_pos[first.key] = first
        self.key_pos[node.key] = node
        return node.val

    def put(self, key: int, value: int) -> None:
        if key not in self.key_pos:
            if self.capacity > 0:
                self.capacity -= 1
            else:
                self.key_pos.pop(self.tail.pre.key)
                lastlast = self.tail.pre.pre
                self.tail.pre = lastlast
                lastlast.nxt = self.tail
            node = DoubleLinkedNode(key, value)
        else:
            node = self.key_pos[key]
            node.val = value
            node.pre.nxt = node.nxt
            node.nxt.pre = node.pre
        first = self.head.nxt
        self.head.nxt = node
        node.pre = self.head
        node.nxt = first
        first.pre = node
        self.key_pos[first.key] = first
        self.key_pos[node.key] = node

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

# 7. https://leetcode-cn.com/problems/merge-intervals/
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if intervals == []:
            return []
        intervals.sort(key=lambda x: x[0])
        res = []
        for i in intervals:
            if not res or res[-1][1] < i[0]:
                res.append(i)
            else:
                res[-1][1] = max(res[-1][1], i[1])
        return res

# 8. https://leetcode-cn.com/problems/n-queens/description/
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        def recursion(col, pie, na, sub):
            if len(sub) == n:
                res.append(sub[:])
                return
            combine = col | pie | na
            for i in range(n):
                if combine >> i & 1 == 0:
                    sub.append(i)
                    choose = 1 << i
                    recursion(col | choose, (pie | choose) << 1, (na | choose) >> 1, sub)
                    sub.pop()
        col, pie, na = 0, 0, 0
        res = []
        recursion(col, pie, na, [])
        return [["." * j + "Q" + "." * (n-j-1) for j in i]for i in res]

# 9. https://leetcode-cn.com/problems/n-queens-ii/description/
class Solution:
    def totalNQueens(self, n: int) -> int:
        def recursion(count, row_num, col, pie, na):
            if row_num == n:
                return count + 1
            combine = col | pie | na
            for i in range(n):
                if combine >> i & 1 == 0:
                    count = recursion(count, row_num + 1, col | 1 << i, (pie | 1 << i) << 1, (na | 1 << i) >> 1)
            return count

        return recursion(0, 0, 0, 0, 0)

# 10. https://leetcode-cn.com/problems/reverse-pairs/
class Solution:
    def reversePairs(self, nums: List[int]) -> int:
        def mergesort(count, nums, left, right):
            if right <= left:
                return count
            mid = (left + right) >> 1
            count = mergesort(count, nums, left, mid)
            count = mergesort(count, nums, mid + 1, right)
            return merge(count, nums, left, mid, right)
        def merge(count, nums, left, mid, right):
            array = [0] * (right - left + 1)
            i, j = left, mid + 1
            while i <= mid and j <= right:
                if nums[i] / 2.0 > nums[j]:
                    count += mid - i + 1
                    j += 1
                else:
                    i += 1
            p, p1, p2 = 0, left, mid + 1
            while p1 <= mid and p2 <=right:
                if nums[p1] <= nums[p2]:
                    array[p] = nums[p1]
                    p1 += 1
                else:
                    array[p] = nums[p2]
                    p2 += 1
                p += 1
            if p1 <= mid:
                array[p:] = nums[p1:mid+1]
            if p2 <= right:
                array[p:] = nums[p2:right+1]
            nums[left: right+1] = array
            return count
        return mergesort(0, nums, 0, len(nums)-1)