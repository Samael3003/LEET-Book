

## 1710. Maximum Units on a Truck

    class Solution {
        public int maximumUnits(int[][] boxTypes, int truckSize) {
         Arrays.sort(boxTypes, (a, b) -> b[1] - a[1]);
          int ans = 0;
          for(int[] b : boxTypes) {
            int count = Math.min(b[0], truckSize);
            ans += count * b[1];
            truckSize -= count;
            if(truckSize == 0) return ans;
          }
          return ans;
        }
    }
  

## 1465. Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts

    class Solution {
        public int maxArea(int h, int w, int[] horizontalCuts, int[] verticalCuts) 
        {
            Arrays.sort(horizontalCuts);

            Arrays.sort(verticalCuts);


            int max = horizontalCuts[0];

            for(int i=1;i<horizontalCuts.length;i++)

                max = Math.max(horizontalCuts[i]-horizontalCuts[i-1],max);

            max = Math.max(h-horizontalCuts[horizontalCuts.length-1],max);


            int maxv = verticalCuts[0];
            for(int i=1;i<verticalCuts.length;i++)
                maxv = Math.max(verticalCuts[i]-verticalCuts[i-1],maxv);
            maxv = Math.max(w-verticalCuts[verticalCuts.length-1],maxv);


            return (int)(1L*max*maxv %1000000007);
        }
    }


376. Wiggle Subsequence

    class Solution {
        public int wiggleMaxLength(int[] nums) {
            if(nums.length<2)return 1;

            int count=1;

            int prevDiff=0;
            for(int i=1;i<nums.length;i++){
                int diff = nums[i] - nums[i-1];
                if((diff>0 && prevDiff <= 0) || (diff<0 && prevDiff >= 0)){
          // the equals in prevDiff <= 0 && prevDiff >=0 can only be used at the first iteration, otherwise prevDiff will never be zero afterwards
                    count++;
                    prevDiff = diff;
                }
            }
            return count;
        }
    }


## 135. Candy

    class Solution {
        public int candy(int[] ratings) {
            int n =  ratings.length;
            if(n==0)
                return 0;
            int left[] = new int[n];
            Arrays.fill(left,1);
            for(int i=1;i<n;i++)
            {
                if(ratings[i] > ratings[i-1])
                    left[i] = left[i-1]+1;
            }
            int right[] = new int[n];
            Arrays.fill(right,1);
            for(int i=n-2;i>=0;i--)
            {
                if(ratings[i] > ratings[i+1])
                    right[i] = right[i+1]+1;
            }
            int ans = 0;
            for(int i=0;i<n;i++)
                ans += Math.max(left[i],right[i]);
            return ans;
        }
    }



## 128. Longest Consecutive Sequence

    class Solution {
        public int longestConsecutive(int[] nums) {
          Set <Integer> set = new HashSet <Integer>();
          for(int n : nums)
            set.add(n);

          int streak = 0;
          for(int n : nums) {
            if(!set.contains(n - 1)) {
              int currNum = n;
              int currStreak = 1;

              while(set.contains(currNum + 1)) {
                currNum ++;
                currStreak ++;
              }

              streak = Math.max(streak , currStreak);
          }
        }
          return streak;
      }
    }




## 509. Fibonacci Number

    class Solution {
        public int fib(int n) {

            return (int)((Math.pow(((1+Math.sqrt(5))/2),n) - Math.pow(((1-Math.sqrt(5))/2),n))/Math.sqrt(5));
        }

    }



## 97. Interleaving String

    class Solution {
    public boolean isInterleave(String s1, String s2, String s3) {
    int m = s1.length();
    int n = s2.length();

        // Base case
        if(s3.length() != m + n) {
            return false;
        }
        if(s1.length() == 0) {
            return s2.equals(s3);
        } 
        if(s2.length() == 0) {
            return s1.equals(s3);
        }

        // dp[i][j] represents can we use i characters from s1 and j characters from s2 
        // to form the first i+j characters from s3
        boolean[][] dp = new boolean[m+1][n+1];

        // base case using 0 characters from both means yes. 
        dp[0][0] = true;

        for(int i = 1; i < m+1; i++) {
            dp[i][0] = dp[i-1][0] && s1.charAt(i-1) == s3.charAt(i-1);
        }

        for(int i = 1; i < n+1; i++) {
            dp[0][i] = dp[0][i-1] && s2.charAt(i-1) == s3.charAt(i-1);
        }

        // the recursive relationship
        for(int i = 1; i < m+1; i++) {
            for(int j = 1; j < n+1; j++) {
                dp[i][j] = (dp[i-1][j] && s1.charAt(i-1) == s3.charAt(i+j-1)) 
                        || (dp[i][j-1] && s2.charAt(j-1) == s3.charAt(i+j-1));
            }
        }

        return dp[m][n];
    }
    }



1473. Paint House III

    class Solution {
        Integer[][][] dp;
        public int minCost(int[] houses, int[][] cost, int m, int n, int target) {
            dp = new Integer[m][n+1][target+1];

            // start from first house
            // let prev house (before first) color is 0(not already painted)
            // initially no neighbours
            int ans = solve(0,0,0,houses,cost,m,n,target);
            return (ans==Integer.MAX_VALUE/2)?-1:ans;
        }

        /*
            i       : index to traverse houses
            CI      : for color of previous house
            NB      : number of neighbours
        */
        private int solve(int i, int CI, int NB, int[] houses, int[][] cost, int m, int n, int target)
        {
            /* If # of neighbours exceeds the target, then return MAX */
            if(NB>target)
                return Integer.MAX_VALUE/2;

            /* If reached to last house then,
                        - if neighbours reached target then return 0
                        - else return MAX as neighbour are not exactly equal to target
             */
            if(i==m)
                return (NB==target) ? 0 : Integer.MAX_VALUE/2;

            if(dp[i][CI][NB]!=null) return dp[i][CI][NB];

            int ans = Integer.MAX_VALUE/2;

            // If the house is not painted
            if(houses[i]==0){

                /* then check for the min cost required from all the available cost values
                    i+1 : to move next house
                    j+1 : to check for next cost 
                    if CI==j+1 -> if the house is neighbour then NB else NB+1
                */
                for(int j=0; j<n; j++){
                    ans = Math.min(ans, cost[i][j]+solve(i+1, j+1, (CI==j+1)?NB:NB+1, houses,cost,m,n,target));}
            }

            // If the house is already painted then simply move to next house
            else{
                ans = Math.min(ans,solve(i+1,houses[i], (houses[i]==CI)?NB:NB+1, houses, cost, m, n, target));
            } 

            return dp[i][CI][NB] = ans;

        }
    }



1696. Jump Game VI

    class Solution {

        class Pair{
            int num;
            int index;

            Pair(int num, int index){
                this.num = num;
                this.index = index;
            }
        }

        public int maxResult(int[] nums, int k) {
            int[] dp = new int[nums.length];
            PriorityQueue<Pair> pq = new PriorityQueue<>(new Comparator<Pair>() {
                @Override
                public int compare(Pair o1, Pair o2) {
                    return o2.num - o1.num;
                }
            });

            for(int i = nums.length - 1; i >= 0; i--){
                if(i == nums.length - 1){
                    dp[i] = nums[i];
                    pq.add(new Pair(nums[i], i));
                }else{
                    int start = i+1;
                    int end = Math.min(nums.length - 1, i + k);

                    if(start == end){
                        dp[i] = nums[i] + dp[i+1];
                        pq.add(new Pair(dp[i], i));
                    }else{
                        while (!pq.isEmpty()){
                            Pair p = pq.peek();

                            if(p.index >= start && p.index <= end){
                                dp[i] = nums[i] + p.num;
                                pq.add(new Pair(dp[i], i));
                                break;
                            }else{
                                pq.poll();
                            }
                        }
                    }
                }
            }
            return dp[0];
        }

    }



746. Min Cost Climbing Stairs

    class Solution {
        public int minCostClimbingStairs(int[] cost) {
           for (int i = 2; i < cost.length; i ++) {
                cost[i] += Math.min(cost[i - 1], cost[i - 2]);
            }
            return Math.min(cost[cost.length - 1], cost[cost.length - 2]);
        }
    }
