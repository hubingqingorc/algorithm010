# 1. https://leetcode-cn.com/problems/first-unique-character-in-a-string/
class Solution:
    def firstUniqChar(self, s: str) -> int:
        unique = collections.OrderedDict()
        visit = set()
        for i in range(len(s)):
            if s[i] not in visit:
                if s[i] in unique:
                    unique.pop(s[i])
                    visit.add(s[i])
                else:
                    unique[s[i]] = i
        if unique:
            for k, v in unique.items():
                return v
        return -1

# 2. https://leetcode-cn.com/problems/reverse-string-ii/
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        length = len(s)
        s = list(s)
        reserve = length % (2 * k)
        i, check = 0, length - reserve
        while i < length:
            if i < check or reserve >= k:
                left, right = i, i + k-1
            else:
                left, right = i, i + reserve - 1
            while left < right:
                s[left], s[right] = s[right], s[left]
                left += 1
                right -= 1
            i += 2 * k
        return "".join(s)

# 3. https://leetcode-cn.com/problems/reverse-words-in-a-string/
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(reversed(s.split()))

# 4. https://leetcode-cn.com/problems/reverse-words-in-a-string-iii/
class Solution:
    def reverseWords(self, s: str) -> str:
        return " ".join(["".join(reversed(list(i))) for i in s.split()])

# 5. https://leetcode-cn.com/problems/reverse-only-letters/
class Solution:
    def reverseOnlyLetters(self, S: str) -> str:
        upper_l, upper_r = ord("A"), ord("Z")
        lower_l, lower_r = ord("a"), ord("z")
        c_list = []
        for i in S:
            if upper_l <= ord(i) <= upper_r or lower_l <= ord(i) <= lower_r:
                c_list.append(i)
        l, r = 0, len(c_list)-1
        while l < r:
            c_list[l], c_list[r] = c_list[r], c_list[l]
            l += 1
            r -= 1
        res, p = [], 0
        for i in S:
            if upper_l <= ord(i) <= upper_r or lower_l <= ord(i) <= lower_r:
                res.append(c_list[p])
                p += 1
            else:
                res.append(i)
        return "".join(res)

# 6. https://leetcode-cn.com/problems/isomorphic-strings/
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        s_t, t_s = dict(), dict()
        for i in range(len(s)):
            if s[i] not in s_t:
                s_t[s[i]] = t[i]
            else:
                if s_t[s[i]] != t[i]:
                    return False
            if t[i] not in t_s:
                t_s[t[i]] = s[i]
            else:
                if t_s[t[i]] != s[i]:
                    return False
        return True

# 7. https://leetcode-cn.com/problems/valid-palindrome-ii/
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def check(l, r):
            while l <= r:
                if s[l] != s[r]:
                    return False
                l += 1
                r -= 1
            return True
        s = list(s)
        l, r = 0, len(s)-1
        while l <= r:
            if s[l] == s[r]:
                l += 1
                r -= 1
            else:
                return check(l+1, r) or check(l, r-1)
        return True

# 8. https://leetcode-cn.com/problems/longest-increasing-subsequence/
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        def binary_search_insert(array, left, right, insert):
            if right - left <= 1:
                if insert < array[left]:
                    array[left] = insert
                elif insert > array[left]:
                    array[right] = insert
                return
            mid = (left + right) >> 1
            if insert == array[mid]:
                return
            elif insert > array[mid]:
                binary_search_insert(array, mid, right, insert)
            else:
                binary_search_insert(array, left, mid, insert)
            return
        dp = [nums[0]]
        for i in range(1, len(nums)):
            if nums[i] > dp[-1]:
                dp.append(nums[i])
            else:
                binary_search_insert(dp, 0, len(dp)-1, nums[i])
        return len(dp)

# 9. https://leetcode-cn.com/problems/decode-ways/
class Solution:
    def numDecodings(self, s: str) -> int:
        if len(s) == 0 or s[0] == "0":
            return 0
        dp = [0] * (len(s) + 1)
        dp[0], dp[1] = 1, 1
        for i in range(1, len(s)):
            if s[i] == "0":
                if s[i-1] == "1" or s[i-1] == "2":
                    dp[i+1] = dp[i-1]
                else:
                    return 0
            else:
                if int (s[i-1:i+1]) > 26 or s[i-1] == "0":
                    dp[i+1] = dp[i]
                else:
                    dp[i+1] = dp[i] + dp[i-1]
        return dp[-1]

# 10. https://leetcode-cn.com/problems/string-to-integer-atoi/
class Solution:
    def myAtoi(self, str: str) -> int:
        res, pos, length = 0, 0, len(str)
        # 跳过空格
        while pos < length and str[pos] == " ":
            pos += 1
            # 确定符号
        base = ord("0")
        if pos < length:
            if str[pos] == "+":
                sign = 1
                pos += 1
            elif 0 <= ord(str[pos]) - base <= 9:
                sign = 1
            elif str[pos] == "-":
                sign = -1
                pos += 1
            else:
                return res
        # 处理数字位
        D, data = 10, 0
        if pos < length:
            bit = ord(str[pos]) - base
            while 0 <= bit <= 9:
                data = D * data + bit
                pos += 1
                if pos < length:
                    bit = ord(str[pos]) - base
                else:
                    break
            if sign == 1:
                res = min(2 ** 31 - 1, data)
            else:
                res = max(-2 ** 31, -1 * data)

        return res


# 11. https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        len_p, len_s, count = len(p), len(s), {}
        if len_s < len_p:
            return []
        for c in p:
            count[c] = 1 if c not in count else count[c] + 1
        res = []
        for i in range(len_p):
            if s[i] in count:
                count[s[i]] -= 1
        ok = [0] * len(list(count.values()))
        if list(count.values()) == ok:
            res.append(0)

        for i in range(len_p, len_s):
            if s[i - len_p] in count:
                count[s[i - len_p]] += 1
            if s[i] in count:
                count[s[i]] -= 1
                if count[s[i]] == 0 and list(count.values()) == ok:
                    res.append(i - len_p + 1)
        return res

# 12. https://leetcode-cn.com/problems/longest-palindromic-substring/
class Solution:
    def longestPalindrome(self, s: str) -> str:
        length = len(s)
        if length == 0:
            return ""
        s = list(s)
        count, l_r = 1, [0,0]
        def find(l, r, sub_count, count, l_r):
            while l >= 0 and r < length and s[l] == s[r]:
                sub_count += 2
                if sub_count > count:
                    count = sub_count
                    l_r = [l,r]
                l -= 1
                r += 1
            return count, l_r
        for i in range(length):
            count, l_r = find(i-1, i+1, 1, count, l_r)
            count, l_r = find(i, i+1, 0, count, l_r)
        return "".join(s[l_r[0]: l_r[1]+1])

# 13.https://leetcode-cn.com/problems/longest-valid-parentheses/
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        dp = [0] * len(s)
        res = 0
        for i in range(1, len(s)):
            if s[i] == ")":
                right, left = i, i - dp[i-1] - 1
                if left >= 0 and s[left] == "(":
                    dp[i] = dp[i-1] + 2 + dp[left-1]
            res = max(res, dp[i])
        return res

# 14. https://leetcode-cn.com/problems/race-car/
class Solution:
    def racecar(self, target: int) -> int:
        dp = [0, 1, 4] + [math.inf] * target
        for i in range(3, target + 1):
            b = len(bin(i)) - 2
            if 2 ** b - 1 == i:
                dp[i] = b
                continue
            for j in range(b - 1):
                dp[i] = min(dp[i], b-1 + j+2 + dp[i - 2**(b-1) + 2**j])
            dp[i] = min(dp[i], b + 1 + dp[2**b - 1 - i])
        return dp[target]

# 15. https://leetcode-cn.com/problems/wildcard-matching/
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        len_s, len_p = len(s), len(p)
        dp = [[False] * (len_s + 1) for _ in range(len_p + 1)]
        dp[0][0] = True
        k = 0
        while k < len_p and p[k] == "*":
            dp[k + 1][0] = True
            k += 1
        for i in range(len_p):
            if p[i] == "*":
                for j in range(len_s):
                    dp[i+1][j+1] = dp[i][j] or dp[i][j+1] or dp[i+1][j]
            elif p[i] == "?":
                for j in range(len_s):
                    dp[i+1][j+1] = dp[i][j]
            else:
                for j in range(len_s):
                    if p[i] == s[j]:
                        dp[i+1][j+1] = dp[i][j]
        return dp[-1][-1]

# 16. https://leetcode-cn.com/problems/distinct-subsequences/
class Solution:
    def numDistinct(self, s: str, t: str) -> int:
        len_s, len_t = len(s), len(t)
        if len_s < len_t:
            return 0
        dp = [[0] * (len_s + 1) for _ in range(len_t + 1)]
        dp[0] = [1] * (len_s + 1)
        for i in range(len_t):
            for j in range(len_s):
                dp[i+1][j+1] = dp[i][j] + dp[i+1][j] if s[j] == t[i] else dp[i+1][j]
        return dp[-1][-1]