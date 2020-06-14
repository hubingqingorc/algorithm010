# -*- coding: utf-8 -*-
# 1. 删除排序数组中的重复项
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        """
        1.暴力法

        i = 0
        while i < len(nums) - 1:
            j = i + 1
            while j < len(nums):
                if nums[i] == nums[j]:
                    del nums[j]
                else:
                    break
            i += 1
        return len(nums)

        2.双指针
        """
        p = 0
        for num in nums[1:]:
            if nums[p] != num:
                p += 1
                nums[p] = num
        return p + 1

# 2.旋转数组
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        """
        Do not return anything, modify nums in-place instead.
        1.暴力法

        length = len(nums)
        k = k % length
        for i in range(length - k - 1, -1, -1):
            j = i
            for _ in range(k):
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
                j += 1

        2.整体移动

        2.1

        length = len(nums)
        k = k % length
        if k > 0:
            if k < length / 2:
                data = nums[-k:]
                nums[k:] = nums[:length-k]
                nums[:k] = data
            else:
                data = nums[:length-k]
                nums[:k] = nums[-k:]
                nums[-(length-k):] = data

        2.2

        length = len(nums)
        k = k % length
        nums[:] = nums[length-k:] + nums[:length-k]

        3.环状代替

        length = len(nums)
        k = k % length
        count = 1
        for start in range(k):
            cur = start
            nxt = (cur + k) % length
            temp1 = nums[cur]
            temp2 = nums[nxt]
            nums[nxt] = temp1
            count += 1
            while nxt != start:
                cur = nxt
                nxt = (cur + k) % length
                temp1 = temp2
                temp2 = nums[nxt]
                nums[nxt] = temp1
                count += 1
            if count >= length:
                break

        4.使用反转

        length = len(nums)
        k = k % length
        for i in range((length - k) //2):
            nums[i], nums[length - k - i - 1] = nums[length - k - i - 1], nums[i]
        for i in range(k // 2):
            nums[length - k + i], nums[length - i - 1] = nums[length - i - 1], nums[length - k + i]
        for i in range(length // 2):
            nums[i], nums[length - 1 - i] = nums[length - 1 - i], nums[i]

        """
        length = len(nums)
        k = k % length
        nums[:] = nums[length-k:] + nums[:length-k]

# 3.合并两个有序链表
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        """
        1.暴力法

        prehead = ListNode(-1)
        head = prehead
        if l1 is None:
            prehead.next = l2
        elif l2 is None:
            prehead.next = l1
        else:
            while l1 is not None and l2 is not None:
                if l1.val <= l2.val:
                    head.next = l1
                    l1 = l1.next
                    head = head.next
                else:
                    head.next = l2
                    l2 = l2.next
                    head = head.next
        if l1 is None:
            head.next = l2
        elif l2 is None:
            head.next = l1
        return prehead.next

        2.递归法

        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

        """
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        if l1.val <= l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

# 4.合并两个有序数组
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        1.暴力法

        nums1[m:m+n] = nums2
        for i in range(m, m+n):
            j = i - 1
            if nums1[j] <= nums1[i]:
                break
            while j >= 0:
                if nums1[j] <= nums1[j+1]:
                    break
                else:
                    nums1[j],nums1[j+1] = nums1[j+1], nums1[j]
                    j -= 1
        (借助sorted)
        nums1[m:m+n] = nums2
        nums1[:m + n] = sorted(nums1[:m+n])

        2.双指针
        (从小到大)
        num_copy = nums1[:m]
        p, p1, p2 = 0, 0, 0
        while p1 < m and p2 < n:
            if num_copy[p1] <= nums2[p2]:
                nums1[p1 + p2] = num_copy[p1]
                p1 += 1
            else:
                nums1[p1 + p2] = nums2[p2]
                p2 += 1
        if p1 == m:
            nums1[p1 + p2:]=nums2[p2:]
        elif p2 == n:
            nums1[p1 + p2:]=num_copy[p1:]

        （从大到小）
        p1, p2 = m - 1, n - 1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] >= nums2[p2]:
                nums1[p1 + p2 + 1] = nums1[p1]
                p1 -= 1
            else:
                nums1[p1 + p2 + 1] = nums2[p2]
                p2 -= 1
        if p1 < 0:
            nums1[:p1 + p2 + 2] = nums2[:p2 + 1]

        """
        p1, p2, p = m - 1, n - 1, m + n - 1
        while p1 >= 0 and p2 >= 0:
            if nums1[p1] >= nums2[p2]:
                nums1[p] = nums1[p1]
                p1 -= 1
                p -= 1
            else:
                nums1[p] = nums2[p2]
                p2 -= 1
                p -= 1
        if p1 < 0:
            nums1[:p + 1] = nums2[:p2 + 1]

# 5.两数之和
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        1.暴力法

        for i in range(len(nums) - 1):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]

        2.粗筛 + 精拣

        for i in range(len(nums) - 1):
            d_val = target - nums[i]
            if d_val in nums[i+1:]:
                return [i, i + 1 + nums[i+1:].index(d_val)]

        3.哈希（字典模拟）
        """
        num_idx = {}
        for i, num in enumerate(nums):
            if target - num in num_idx:
                return [num_idx[target - num], i]
            else:
                num_idx[num] = i
# 6.移动零
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        1.暴力法
        length = len(nums)
        for i in range(length - 1, -1, -1):
            if nums[i] == 0:
                j = i + 1
                while j < length and nums[j] != 0:
                    nums[j], nums[j-1] = nums[j-1], nums[j]
                    j += 1
        2.删零补零
        count_zero = 0
        i = 0
        while i < len(nums):
            if nums[i] == 0:
                count_zero += 1
                del nums[i]
            else:
                i += 1
        for _ in range(count_zero):
            nums.append(0)
        3.双指针
        """
        j = 0
        length = len(nums)
        for i in range(length):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1
        while j < length:
            nums[j] = 0
            j += 1

# 7.加一
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        """
        1.暴力法

        val = 0
        length = len(digits)
        for i in range(length):
            val += digits[i] * 10**(length-1-i)
        val += 1
        if val // 10**(length-1) < 10:
            for j in range(length):
                digits[j] = val // 10**(length-1-j)
                val = val % 10**(length-1-j)
        else:
            for j in range(length):
                digits[j] = val // 10**(length-j)
                val = val % 10**(length-j)
            digits.append(val%10)
        return digits

        2.进位
        """
        if digits[0] == 0:
            return [1]
        else:
            for i in range(len(digits)-1, -1, -1):
                if digits[i] != 9:
                    digits[i] += 1
                    return digits
                else:
                    digits[i] = 0
            if digits[0] == 0:
                digits[0] = 1
                digits.append(0)
                return digits
# 8.设计循环双端队列
class MyCircularDeque:

    def __init__(self, k: int):
        """
        Initialize your data structure here. Set the size of the deque to be k.
        1.双指针循环
        """
        self.que = [0 for _ in range(k)]
        self.front = 0
        self.rear = 1
        self.capacity = k

    def insertFront(self, value: int) -> bool:
        """
        Adds an item at the front of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        else:
            self.que[self.front % self.capacity] = value
            self.front -= 1
            return True

    def insertLast(self, value: int) -> bool:
        """
        Adds an item at the rear of Deque. Return true if the operation is successful.
        """
        if self.isFull():
            return False
        else:
            self.que[self.rear % self.capacity] = value
            self.rear += 1
            return True

    def deleteFront(self) -> bool:
        """
        Deletes an item from the front of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        else:
            self.front += 1
            return True

    def deleteLast(self) -> bool:
        """
        Deletes an item from the rear of Deque. Return true if the operation is successful.
        """
        if self.isEmpty():
            return False
        else:
            self.rear -= 1
            return True

    def getFront(self) -> int:
        """
        Get the front item from the deque.
        """
        if self.isEmpty():
            return -1
        else:
            return self.que[(self.front + 1) % self.capacity]

    def getRear(self) -> int:
        """
        Get the last item from the deque.
        """
        if self.isEmpty():
            return -1
        else:
            return self.que[(self.rear - 1) % self.capacity]

    def isEmpty(self) -> bool:
        """
        Checks whether the circular deque is empty or not.
        """
        if self.front == self.rear - 1:
            return True
        else:
            return False

    def isFull(self) -> bool:
        """
        Checks whether the circular deque is full or not.
        """
        if (self.rear - self.front - 1) == self.capacity:
            return True
        else:
            return False

# 9.接雨水
class Solution:
    def trap(self, height: List[int]) -> int:
        """
        1.暴力法

        rain = 0
        length = len(height)
        for i in range(1, length-1):
            l_max, r_max = 0, 0
            for j in range(i-1, -1, -1):
                l_max = max(l_max, height[j])
            for k in range(i+1, length):
                r_max = max(r_max, height[k])
            if min(l_max, r_max) > height[i]:
                rain += min(l_max, r_max) - height[i]
        return rain

        2.双指针

        res = 0
        length = len(height)
        if length == 0:
            return 0
        l, r = 0, length - 1
        max_l, max_r = height[l], height[r]
        while l < r:
            max_l = max(max_l, height[l])
            max_r = max(max_r, height[r])
            if max_l < max_r:
                res += max_l - height[l]
                l += 1
            else:
                res += max_r - height[r]
                r -= 1
        return res

        3.栈
        """
        stack = []
        res = 0
        for i in range(len(height)):
            while stack != [] and height[i] > height[stack[-1]]:
                cur = stack[-1]
                stack.pop()
                if stack == []:
                    break
                res += (min(height[i], height[stack[-1]]) - height[cur]) * (i - stack[-1] - 1)
            stack.append(i)
        return res

