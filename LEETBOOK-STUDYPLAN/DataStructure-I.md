# DAY - I

### 217. Contains Duplicate
        class Solution:
            def containsDuplicate(self, nums: List[int]) -> bool:
                nums = sorted(nums)
                for i in range (len(nums)-1):
                    if nums[i] == nums[i + 1]:
                        return True
                return False


### 53. Maximum Subarray
        class Solution(object):
            def maxSubArray(self, nums):
                 maxSum = float('-inf')
                 currSum = 0
                 for n in range(len(nums)):
                     if currSum < 0:
                         currSum = 0
                     currSum += nums[n]
                     maxSum = max(maxSum,currSum)
                 return maxSum

# DAY - II

### 1. Two Sum
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = {}

        for i in range(0, len(nums)):

            if target - nums[i] in d:
                return [d[target-nums[i]], i]
            else:
                d[nums[i]] = i
                
### 88. Merge Sorted Array 
        >IN JAVASCRIPT
        
        function merge(targetArray, firstLength, fillerArray, secondLength) {
            let left = firstLength - 1;
            let right = secondLength - 1;
            let reversedIndex = firstLength + secondLength - 1;

            while(left >= 0 && right >= 0) {
                // Pick the bigger number from either of the arrays and reduce it's corresponding index
                targetArray[reversedIndex--] = targetArray[left] > fillerArray[right] ? targetArray[left--] : fillerArray[right--];
            }

            // Fill leftover sorted values from second array
            while(right >= 0) {
                targetArray[reversedIndex--] = fillerArray[right--];
            }

            return targetArray;
        };


# DAY - III

###  349. Intersection of Two Arrays
        class Solution(object):
            def intersection(self, nums1, nums2):
                """
                :type nums1: List[int]
                :type nums2: List[int]
                :rtype: List[int]
                """
                set1 = set(nums1)
                set2 = set(nums2)
                result = []

                for num in set1:
                    if num in set2:
                        result.append(num)

                return result
        
### 121. Best Time to Buy and Sell Stock
        class Solution:
            def maxProfit(self, prices: List[int]) -> int:
                left = 0 #Buy
                right = 1 #Sell
                max_profit = 0
                while right < len(prices):
                    currentProfit = prices[right] - prices[left] #our current Profit
                    if prices[left] < prices[right]:
                        max_profit =max(currentProfit,max_profit)
                    else:
                        left = right
                    right += 1
                return max_profit
   
   
# DAY - IV

### 566. Reshape the Matrix        
        class Solution:
            def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:

                m = len(mat)
                n = len(mat[0])

                save = []
                for i in mat:       
                    save = save + i

                if m * n != r * c:
                    return mat
                else:
                    save1 = []
                    for i in range(r):
                        save1.append(save[i*c:i*c+c])

                return save1        
        
### 118. Pascal's Triangle     
        class Solution:
            def generate(self, numRows: int) -> List[List[int]]:
                triangle = [[1]]
                for _ in range(numRows - 1):
                    triangle.append([1])
                    for i in range(len(triangle[-2]) - 1):
                        triangle[-1].append(triangle[-2][i] + triangle[-2][i+1])
                    triangle[-1].append(1)
                return triangle        
