# DAY - 1

### 34. Find First and Last Position of Element in Sorted Array
        class Solution:
            def searchRange(self, nums: List[int], target: int) -> List[int]:
                leftIdx = bisect.bisect_left(nums, target)
                if not 0 <= leftIdx < len(nums) or nums[leftIdx] != target:
                    return [-1, -1]
                return [leftIdx, bisect.bisect_right(nums, target) - 1]
            # Time: O(log(n))
            # Space: O(1)

### 33. Search in Rotated Sorted Array
        class Solution:
            def search(self, nums: List[int], target: int) -> int:
                low = 0
                high = len(nums)-1
                while low <= high:
                    mid = (high+low)//2
                    if nums[mid] == target:
                        return mid
                    elif nums[low] <= nums[mid]:
                        # low to mid id sorted
                        if target >= nums[low] and target < nums[mid]:
                            high = mid-1
                        else:
                            low = mid+1
                    elif nums[mid] <= nums[high]:
                        # mid to high part is sorted
                        if target > nums[mid] and target <= nums[high]:
                            low = mid+1
                        else:
                            high = mid-1
                return -1
        
        
### 74. Search a 2D Matrix        
        class Solution:
            def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
                rows = len(matrix)
                cols = len(matrix[0])
                l, r = 0, rows * cols - 1

                while l < r:
                    mid = (l + r) >> 1
                    elem = matrix[mid // cols][mid % cols]
                    if elem == target:
                        return True
                    if elem < target:
                        l = mid + 1
                    else:
                        r = mid - 1

                return matrix[l // cols][l % cols] == target
        
# DAY -2

### 153. Find Minimum in Rotated Sorted Array        
        class Solution(object):
            def findMin(self, nums):
                L, R = 0, len(nums) - 1

                while L < R:
                    m = (L + R) // 2
                    if nums[m] > nums[R]:
                        L = m + 1
                    else:
                        R = m

                return nums[R]
        
### 162. Find Peak Element        
        class Solution:
            def findPeakElement(self, nums: List[int]) -> int:
                N = len(nums)
                l , r = 1 , len(nums)-2 # we'll not search first & last element

                if N == 1: return 0 # edge case [1]
                if N > 1 and nums[0] > nums[1]: return 0 # checking first element
                if N > 1 and nums[-1] > nums[-2]: return N-1 # checking last element

                while l <= r:
                    m = (l+r) // 2

                    if nums[m] >= nums[m-1] and nums[m] >= nums[m+1]: return m
                    elif nums[m-1] >= nums[m]: r = m-1
                    else: l = m+1

                return -1
