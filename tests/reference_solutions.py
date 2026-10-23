"""仅用于评估器自检的自有参考实现；不放入模型提示或训练数据。"""

from collections import Counter
from itertools import combinations, product
from math import isqrt


class Solution:
    def isPalindrome(self, x):
        return str(x) == str(x)[::-1]

    def maxArea(self, height):
        return max((j-i)*min(height[i], height[j]) for i, j in combinations(range(len(height)), 2))

    def threeSum(self, nums):
        return [list(t) for t in sorted({tuple(sorted(t)) for t in combinations(nums, 3) if sum(t) == 0})]

    def letterCombinations(self, digits):
        mapping = dict(zip("23456789", ["abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]))
        return ["".join(t) for t in product(*(mapping[d] for d in digits))] if digits else []

    def generateParenthesis(self, n):
        result = []
        for chars in product("()", repeat=2*n):
            balance = 0
            for ch in chars:
                balance += 1 if ch == "(" else -1
                if balance < 0:
                    break
            else:
                if balance == 0:
                    result.append("".join(chars))
        return result

    def isValidSudoku(self, board):
        units = list(board) + list(zip(*board))
        units += [[board[r+i][c+j] for i in range(3) for j in range(3)] for r in (0, 3, 6) for c in (0, 3, 6)]
        return all(len([x for x in unit if x != "."]) == len({x for x in unit if x != "."}) for unit in units)

    def lengthOfLastWord(self, s):
        return len(s.split()[-1])

    def plusOne(self, digits):
        return [int(x) for x in str(int("".join(map(str, digits))) + 1)]

    def addBinary(self, a, b):
        return bin(int(a, 2) + int(b, 2))[2:]

    def mySqrt(self, x):
        return isqrt(x)

    def maxProfit(self, prices):
        return max([0] + [prices[j]-prices[i] for i, j in combinations(range(len(prices)), 2)])

    def singleNumber(self, nums):
        return next(x for x, count in Counter(nums).items() if count == 1)

    def majorityElement(self, nums):
        return Counter(nums).most_common(1)[0][0]

    def isHappy(self, n):
        seen = set()
        while n != 1 and n not in seen:
            seen.add(n)
            n = sum(int(x)**2 for x in str(n))
        return n == 1

    def isIsomorphic(self, s, t):
        return len(set(s)) == len(set(t)) == len(set(zip(s, t)))

    def containsDuplicate(self, nums):
        return len(nums) != len(set(nums))

    def containsNearbyDuplicate(self, nums, k):
        return any(nums[i] == nums[j] and j-i <= k for i, j in combinations(range(len(nums)), 2))

    def summaryRanges(self, nums):
        result = []
        i = 0
        while i < len(nums):
            j = i
            while j+1 < len(nums) and nums[j+1] == nums[j]+1:
                j += 1
            result.append(str(nums[i]) if i == j else str(nums[i]) + "->" + str(nums[j]))
            i = j+1
        return result

    def missingNumber(self, nums):
        return next(x for x in range(len(nums)+1) if x not in nums)

    def canConstruct(self, ransomNote, magazine):
        return not (Counter(ransomNote)-Counter(magazine))
