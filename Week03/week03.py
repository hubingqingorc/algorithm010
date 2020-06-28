# 1.二叉树的最近公共祖先
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        """
        1.递归

        self.ans = None
        def recursion(root, p, q):
            if self.ans is not None:
                return
            left, right = False, False
            if root.left is not None:
                left = recursion(root.left, p, q)
            if root.right is not None:
                right = recursion(root.right, p, q)
            if left and right:
                self.ans = root
                return
            elif root.val == p.val:
                if left or right == True:
                    self.ans = p
                return True
            elif root.val == q.val:
                if left or right == True:
                    self.ans = q
                return True
            return left or right
        recursion(root, p, q)
        return self.ans

        2.字典记录父节点
        """
        if not root or root == p or root == q:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if not left:
            return right
        if not right:
            return left
        return root

# 2.从前序与中序遍历序列构造二叉树
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        if not (preorder and inorder):
            return None
        root = TreeNode(preorder[0])
        idx = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:1+idx], inorder[:idx])
        root.right = self.buildTree(preorder[1+idx:], inorder[idx+1:])
        return root


# 3.组合
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        output = []
        def backtrace(start=1, temp=[]):
            if len(temp) == k:
                output.append(temp[:])
            for i in range(start, n+1):
                temp.append(i)
                backtrace(start=i + 1, temp=temp)
                temp.pop()
        backtrace()
        return output

# 4.全排列
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        output = []
        length = len(nums)
        def backtrace(n):
            if n == length:
                output.append(nums[:])
            for i in range(n, length):
                nums[n], nums[i] = nums[i], nums[n]
                backtrace(n + 1)
                nums[n], nums[i] = nums[i], nums[n]
        backtrace(0)
        return output
