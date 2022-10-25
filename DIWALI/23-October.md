

## 2446. Determine if Two Events Have Conflict

    class Solution {
        public boolean haveConflict(String[] e1, String[] e2) {
            return e1[0].compareTo(e2[1]) <= 0 && e2[0].compareTo(e1[1]) <= 0;
        }
    }
  



## 2447. Number of Subarrays With GCD Equal to K

    class Solution {
        private int gcd(int a, int b) {
            if (b == 0)
                return a;
            return gcd(b, a % b);
        }

        public int subarrayGCD(int[] nums, int k) {
            int ans = 0;
            for (int i = 0; i < nums.length; i++) {
                int currGcd = nums[i];
                if(currGcd == k) // if element is equal to k, increment answer
                    ans++;
                for (int j = i + 1; j < nums.length; j++) {
                    if(nums[j] < k) // if nums[j] < k gcd can never be equal to k for this subarray
                        break;
                    currGcd = gcd(nums[j], currGcd);
                    if (currGcd == k)
                        ans++;
                }
            }
            return ans;
        }
    }
  


## 2448. Minimum Cost to Make Array Equal

    class Solution {
        private long findCost(int[] nums, int[] cost, long x) {
            long res = 0L;
            for (int i = 0; i < nums.length; i++){
                res += Math.abs(nums[i] - x) * cost[i];
            }
            return res;
        }
        public long minCost(int[] nums, int[] cost) {
            long left = 1L;
            long right = 1000000L;
            for (int num : nums) {
                left = Math.min(num, left);
                right = Math.max(num, right);
            }
            long ans = findCost(nums, cost, 1);
            while (left < right) {
                long mid = (left + right) / 2;
                long y1 = findCost(nums, cost, mid);
                long y2 = findCost(nums, cost, mid + 1);
                ans = Math.min(y1, y2);
                if (y1 < y2){
                    right = mid;
                }
                else{
                    left = mid + 1;
                }
            }
            return ans;
        }

    }
  



## 2449. Minimum Number of Operations to Make Arrays Similar

    class Solution {
        public long makeSimilar(int[] nums, int[] target) {
            Arrays.sort(nums); Arrays.sort(target);
            List<Integer> odd_nums = new ArrayList(), even_nums = new ArrayList(), 
            odd_tar = new ArrayList(), even_tar = new ArrayList();
            for (int x: nums) {
                if (x % 2 == 1) odd_nums.add(x);
                else even_nums.add(x);
            }
            for (int x: target) {
                if (x % 2 == 1) odd_tar.add(x);
                else even_tar.add(x);
            }
            long ans = 0;
            for (int i = 0; i < odd_nums.size(); i++)
              if (odd_nums.get(i) > odd_tar.get(i)) 
                  ans += (odd_nums.get(i) - odd_tar.get(i)) / 2;
            for (int i = 0; i < even_nums.size(); i++)
              if (even_nums.get(i) > even_tar.get(i)) 
                  ans += (even_nums.get(i) - even_tar.get(i)) / 2;
            return ans;
        }
    }
