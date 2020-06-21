# 1. 有效的字母异位词
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        """
        1.暴力法

        return sorted(s) == sorted(t)

        2.计数法
        2.1 使用字典
        dic = {}
        for i in s:
            if i in dic:
                dic[i] += 1
            else:
                dic[i] = 1
        for i in t:
            if i in dic:
                dic[i] -= 1
            else:
                return False

        for _, v in dic.items():
            if v != 0:
                return False
        return True

        2.2 使用数组
        """
        if len(s) != len(t):
            return False
        else:
            count = [0] * 26
            for i in range(len(s)):
                count[ord(s[i]) - ord('a')] += 1
                count[ord(t[i]) - ord('a')] -= 1
            for j in count:
                if j != 0:
                    return False
            return True

# 2. 两数之和
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        """
        1.暴力法

        length = len(nums)
        for i in range(length - 1):
            ans = target - nums[i]
            for j in range(i+1, length):
                if ans == nums[j]:
                    return [i, j]

        2.粗筛 + 精拣

        for i in range(len(nums) - 1):
            res = target - nums[i]
            if res in nums[i+1:]:
                return [i, i + 1 + nums[i+1:].index(res)]

        3.哈希(字典)
        """
        num_idx = {}
        for idx, num in enumerate(nums):
            sub = target - num
            if sub in num_idx:
                return [num_idx[sub], idx]
            else:
                num_idx[num] = idx

# 3. N叉树的前序遍历
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""


class Solution:
    """
    1.递归

    def preorder(self, root: 'Node') -> List[int]:
        if root:
            return self.recursion(root)
        else:
            return []

    def recursion(self, root):
        output = [root.val]
        for c in root.children:
            output.extend(self.recursion(c))
        return output

    2.迭代
    """

    def preorder(self, root: 'Node') -> List[int]:
        if root:
            output = []
            stack = [root]
            while len(stack) > 0:
                current = stack.pop()
                output.append(current.val)
                for c in current.children[::-1]:
                    stack.append(c)
            return output
        else:
            return []

# 4.字母异位词分组
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        """
        1.暴力法

        res = []
        length = len(strs)
        label = [0] * length
        strs_sorted = []
        strs_len = []
        for i in range(length):
            strs_sorted.append(sorted(strs[i]))
            strs_len.append(len(strs[i]))
        for i in range(length):
            if label[i] == 0:
                label[i] == 1
                sub_res = [strs[i]]
                s_sorted = strs_sorted[i]
                s_len = strs_len[i]
                for j in range(i + 1, length):
                    if label[j] == 0 and s_len == strs_len[j] and s_sorted == strs_sorted[j]:
                        sub_res.append(strs[j])
                        label[j] = 1
                res.append(sub_res)
        return res

        2.1 按排序分类

        res = {}
        for s in strs:
            s_sorted = tuple(sorted(s))
            if s_sorted in res:
                res[s_sorted].append(s)
            else:
                res[s_sorted] = [s]
        return list(res.values())

        2.2 按字母计数分类
        """
        res = {}
        ord_base = ord('a')
        for s in strs:
            count = [0] * 26
            for c in s:
                count[ord(c) - ord_base] += 1
            key_count = tuple(count)
            if key_count in res:
                res[key_count].append(s)
            else:
                res[key_count] = [s]
        return list(res.values())

# 5. 二叉树的中序遍历
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    """
    1.递归

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root:
            return self.recursion(root)
        else:
            return []

    def recursion(self, root):
        nodes_val = []
        if root.left:
            nodes_val.extend(self.recursion(root.left))
        nodes_val.append(root.val)
        if root.right:
            nodes_val.extend(self.recursion(root.right))
        return nodes_val

    2.迭代（栈）

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        nodes = []
        stack = []
        current = root
        while current != None or len(stack) > 0:
            while current != None:
                stack.append(current)
                current = current.left
            current = stack.pop()
            nodes.append(current.val)
            current = current.right
        return nodes

    3.莫里斯方法
    """

    def inorderTraversal(self, root: TreeNode) -> List[int]:
        nodes = []
        current = root
        while current is not None:
            if current.left is None:
                nodes.append(current.val)
                current = current.right
            else:
                last_pos = current
                current = current.left
                while current.right != None:
                    current = current.right
                current.right = last_pos
                current = last_pos.left
                last_pos.left = None
        return nodes

# 6.二叉树的前序遍历
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    """
    1.递归

    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root:
            return self.recursion(root)
        else:
            return []

    def recursion(self, root):
        vals = []
        vals.append(root.val)
        if root.left:
            vals.extend(self.recursion(root.left))
        if root.right:
            vals.extend(self.recursion(root.right))
        return vals

    2.迭代（栈）

    def preorderTraversal(self, root: TreeNode) -> List[int]:
        if root:
            vals = []
            stack = [root]
            current = root
            while len(stack) > 0:
                current = stack.pop()
                vals.append(current.val)
                if current.right:
                    stack.append(current.right)
                if current.left:
                    stack.append(current.left)
            return vals
        else:
            return []

    3.莫里斯方法
    """
    def preorderTraversal(self, root: TreeNode) -> List[int]:
        output = []
        current = root
        while current:
            output.append(current.val)
            if current.left:
                lastest = current
                current = current.left
                while current.right:
                    current = current.right
                current.right = lastest.right
                current = lastest.left
            else:
                current = current.right
        return output

# 7. N叉树的层序遍历
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    """
    1.递归

    def levelOrder(self, root: 'Node') -> List[List[int]]:
        def recursion(root, lv):
            if len(res) == lv:
                res.append([])
            res[lv].append(root.val)
            for children in root.children:
                recursion(children, lv+1)
        res = []
        if root:
            recursion(root, 0)
        return res


    2.迭代

    def levelOrder(self, root: 'Node') -> List[List[int]]:
        res = []
        if root:
            up_lv = [root]
            while len(up_lv) > 0:
                sub_res = []
                down_lv = []
                for node in up_lv:
                    sub_res.append(node.val)
                    down_lv.extend(node.children)
                res.append(sub_res)
                up_lv = down_lv
        return res

    3.队列

    def levelOrder(self, root: 'Node') -> List[List[int]]:
        res = []
        if root:
            quene = collections.deque([root])
            while quene:
                sub_res = []
                for _ in range(len(quene)):
                    current = quene.popleft()
                    sub_res.append(current.val)
                    quene.extend(current.children)
                res.append(sub_res)
        return res

    """
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        res = []
        if root:
            up_lv = [root]
            while len(up_lv) > 0:
                sub_res = []
                down_lv = []
                for node in up_lv:
                    sub_res.append(node.val)
                    down_lv.extend(node.children)
                res.append(sub_res)
                up_lv = down_lv
        return res

# 8. 丑数
class Solution:
    def nthUglyNumber(self, n: int) -> int:
        """
        1.暴力法
        res = [1]
        i = 2
        while len(res) < n:
            j = i
            # while i != 1 or i != 0:
            while j % 2 == 0:
                j = j / 2
            while j % 5 == 0:
                j = j / 5
            while j % 3 == 0:
                j = j / 3
            if j == 1:
                res.append(i)
            i += 1
        return res[-1]

        2.递推
        """
        res, a, b, c = [1] * n, 0, 0, 0
        for i in range(1, n):
            n2, n3, n5 = res[a] * 2, res[b] * 3, res[c] * 5
            res[i] = min(n2, n3, n5)
            if res[i] == n2:
                a += 1
            if res[i] == n3:
                b += 1
            if res[i] == n5:
                c += 1
        return res[-1]

# 9. 前 K 个高频元素
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        """
        1.哈希 + 堆
        """
        num_count = {}
        for i in nums:
            if i in num_count:
                num_count[i] += 1
            else:
                num_count[i] = 1
        sort_heap = []
        for key, val in num_count.items():
            heapq.heappush(sort_heap, (-val, key))
        res = []
        while len(res) < k:
            res.append(heapq.heappop(sort_heap)[1])
        return res