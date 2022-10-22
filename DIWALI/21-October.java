

## 9. Palindrome Number

    class Solution {
        public boolean isPalindrome(int x) {
            int n = numOfDigits(x);
            int remainder = x;
            int reverse = 0;
            while (remainder > 0) {
                reverse = reverse * 10 +  remainder % 10;
                remainder /= 10;
            }
            return reverse == x;
        }

        int numOfDigits(int x) {
            int i = 1;
            while ((x /= 10) > 0) i++;
            return i;
        }
    }
    

## 2343. Query Kth Smallest Trimmed Number

    class Solution {
        public int[] smallestTrimmedNumbers(String[] nums, int[][] queries) {
            int n = queries.length;
            int l = nums[0].length();
            int z=0;
            int a[]=new int[n];
            for(int i =0;i<n;i++){
                int x = queries[i][1];
                int diff = l-x;
                int t = queries[i][0];
                int y=0;
                List<Pair<String,Integer>> b = new ArrayList<>();
                for(String m : nums){
                    String k = m.substring(diff);
                    b.add(new Pair(k,y));
                     y++;
                    }
                b.sort(Comparator.comparing(Pair::getKey));
                a[z++]=b.get(t-1).getValue();
            }

            return a;
        }
    }
    

## 2342. Max Sum of a Pair With Equal Sum of Digits

    class Solution {
        public int sum(int n){
          int sum = 0;  
        while (n != 0) {  
           sum = sum + n % 10;  
            n = n/10;  
           }
            return sum;
        }
       public int maximumSum(int[] nums) {
            int max =-1;
            HashMap<Integer,PriorityQueue<Integer>> map = new HashMap<>();
          for(int i=0;i<nums.length;i++){
              int num = sum(nums[i]);
               if(map.get(num)==null){
                   map.put(num,new PriorityQueue<Integer>());
               }
              map.get(num).add(nums[i]);
              if(map.get(num).size()==3){
                  map.get(num).poll();
              }
          }
           for(PriorityQueue<Integer> q:map.values()){
               if(q.size()>1){
                   int a=q.poll();
                   int b=q.poll();
                   max=Math.max(max,a+b);
               }
           }     
             return max;
        }
    }
    
    
## 2338. Count the Number of Ideal Arrays

    class Solution {
        public int idealArrays(int n, int m) {
            // step 1 ending number, length, unique plans
            int MOD = (int)1e9 + 7;
            long[][] dp = new long[m + 1][15];

            for(int i = 1; i <= m; i++){
                dp[i][1] = 1;
            }

            // based on current -> calculate next
            // length
            for(int j = 1; j < 14; j++){
                // current state ending
                for(int i = 1; i <= m; i++){
                    // next state ending
                    for(int k = 2; i * k <= m; k++){
                        dp[k * i][j + 1] = (dp[i][j] + dp[k * i][j + 1]) % MOD;
                    }
                }
            }


            // step 2 combination
            long[][] C = new long[n][15];

            for(int i = 0; i < n; i++){
                for(int j = 0; j < 15 && j <= i; j++){
                    if(j == 0) C[i][j] = 1;
                    else C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD;
                }
            }

            // step 3 result 
            long res = 0L;
            for(int i = 1; i <= m; i++){
                for(int j = 1; j <= 14 && j <= n; j++){
                    res = (res + dp[i][j] * C[n - 1][j - 1]) % MOD;
                }
            }

            return (int)res;

        }
    }
    

## 2344. Minimum Deletions to Make Array Divisible**

**PYTHON**

    class Solution:
        def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
            div = gcd(*numsDivide)
            mn = min((n for n in nums if div % n == 0), default = -1)
            return -1 if mn == -1 else sum(n < mn for n in nums)
        # Time: O(m + n) where m and n are the lengths of the two given lists
        # Space: O(1)


## 2348. Number of Zero-Filled Subarrays

    class Solution {
        public long zeroFilledSubarray(int[] nums) {
            ArrayList<Long> list = new ArrayList<>();
            long count=0;
            for(int i=0;i<nums.length;i++){
                if(nums[i]==0)
                    count++;
                else{
                    list.add(count);
                    count=0;
                }
            }
            if(count!=0)
                list.add(count);
            long sum=0;
            for(int i=0;i<list.size();i++)
                sum=sum+(list.get(i)*(list.get(i)+1))/2;
            return sum;
        }
    }

## 2349. Design a Number Container System

    class NumberContainers {
        HashMap<Integer, Integer> map1;
        HashMap<Integer, TreeSet<Integer>> map2;

        public NumberContainers() {
            map1 = new HashMap<>();
            map2 = new HashMap<>();
        }

        public void change(int index, int number) {
            if(map1.containsKey(index) == true){
                int val = map1.get(index);
                TreeSet<Integer> set = map2.get(val);
                set.remove(index);
                if(set.size() == 0){
                    map2.remove(val);
                }
            }
            map1.put(index, number);
            TreeSet<Integer> set = map2.getOrDefault(number, new TreeSet<>());
            set.add(index);
            map2.put(number, set);
        }

        public int find(int number) {
            if(map2.containsKey(number) == true){
                return map2.get(number).first();
            }
            return -1;
        }
    }
    

## 2350. Shortest Impossible Sequence of Rolls

    class Solution {
        public int shortestSequence(int[] rolls, int k) {
            int len = 0;
            Set<Integer> set = new HashSet<>();
            for(int i:rolls)
            {
                set.add(i);
                if(set.size()==k)
                {
                    set = new HashSet<>();
                    len++;
                }
            }
            return len+1;
        }
    }

    // Time Complexity : O(n)

    // Space Complexity : O(k)
    

## 1328. Break a Palindrome

    class Solution {
        public String breakPalindrome(String palindrome) {   
            char ar[]=palindrome.toCharArray();//converts to character array
            if(ar.length<2)//contains the single character
            {
                return "";//returns empty string
            }
            for(int i=0;i<(ar.length/2);i++)//runs upto the half length of array so as to get 
            {                               //lexographically smallest value of string
                if(ar[i]!='a')
                {
                    ar[i]='a';//return the first string replaced by a
                    return String.valueOf(ar);//return string if any character replaced before length/2
                }
            }
            //This condition reaches if all characters withing the array are a 
            //The length has been passed upto the half of the character array

            ar[ar.length-1]='b';
            return String.valueOf(ar);
        }
    }
    

## 237. Delete Node in a Linked List

    class Solution {
        public void deleteNode(ListNode node) {

          node.val = node.next.val;
          node.next = node.next.next;

        }
    }
    

## 16. 3Sum Closest

    class Solution {
        public int threeSumClosest(int[] nums, int target) {
            Arrays.sort(nums);
            // initial closest result is the sum of first 3 values
            int result = nums[0] + nums[1] + nums[2]; 
            for (int i = 0; i < nums.length - 2 && result != target; i++) {
                int val = nums[i];
                int left = i + 1;
                int right = nums.length - 1;
                while (left < right && result != target) {
                    int sum = nums[left] + nums[right];
                    int total = val + sum;
                    if (Math.abs(total - target) < Math.abs(result - target)) {
                        result = total;
                    }

                    if (total == target) break;
                    if (total < target) {
                        left++;
                    } else {
                        right--;
                    }
                }
            }
            return result;
        }
    }


## 653. Two Sum IV - Input is a BST

    class Solution {
        public boolean findTarget(TreeNode root, int k) {
            return findTargetDetail(root, root, k);
        }

        private boolean findTargetDetail(TreeNode root, TreeNode top, int k) {
            if (root == null) return false;

            int target = k - root.val;
            if (root.val != target && findByBinarySearch(top, target)) return true;

            return findTargetDetail(root.left, top, k) || findTargetDetail(root.right, top, k);
        }

        private boolean findByBinarySearch(TreeNode root, int k) {
            if (root == null)
                return false;

            if (root.val == k)
                return true;
            else if (root.val < k)
                return findByBinarySearch(root.right, k);
            else
                return findByBinarySearch(root.left, k);

        }
    }


## 732. My Calendar III

    class MyCalendarThree {
        private TreeMap<Integer, Integer> timeline = new TreeMap<>();
        public int book(int s, int e) {
            timeline.put(s, timeline.getOrDefault(s, 0) + 1); 
            timeline.put(e, timeline.getOrDefault(e, 0) - 1); 
            int ongoing = 0, k = 0;
            for (int v : timeline.values())
                k = Math.max(k, ongoing += v);
            return k;
        }
    }
    

## 623. Add One Row to Tree

    class Solution {
        public TreeNode addOneRow(TreeNode root, int val, int depth) {
            if(depth == 1){
                TreeNode node = new TreeNode(val);
                node.left = root;
                root=node;
            }else{
                Queue<TreeNode> queue = new LinkedList<>();
                queue.add(root);

                while(depth > 0 && !queue.isEmpty()){
                    depth--;
                    int len = queue.size();

                    while(len > 0 && depth>0){
                        TreeNode node = queue.poll();
                        if(node.left!=null) queue.offer(node.left);
                        if(node.right!=null) queue.offer(node.right);
                        if(depth==1){
                            TreeNode nl = new TreeNode(val, node.left,null);
                            node.left=nl;
                            TreeNode rl = new TreeNode(val, null, node.right);
                            node.right=rl;
                        }
                        len--;
                    }
                }
            }
            return root;
        }
    }
    
    
## 112. Path Sum

    class Solution:
        def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
            def dfs(node,currSum) :
                if not node :
                    return False
                currSum +=node.val
                if not node.left and not node.right :
                    return currSum==targetSum
                return (dfs(node.left,currSum)or dfs(node.right,currSum))
            return(dfs(root,0))
            
            
## 2444. Count Subarrays With Fixed Bounds            

    class Solution:
        def countSubarrays(self, nums: List[int], minK: int, maxK: int) -> int:
            n = len(nums) 
            outOfBound = lambda right: nums[right] < minK or nums[right] > maxK
            valid = lambda left, current_max, current_min:left < current_max and left < current_min
            current_min, current_max = -1, -1
            left = -1

            ans = 0        
            for right in range(n):
                if outOfBound(right): 
                    left = right
                    continue               
                current_min = right if nums[right] == minK else current_min
                current_max = right if nums[right] == maxK else current_max

                if valid(left, current_max, current_min):
                    ans += (min(current_max, current_min) - left)

            return ans
            

## 2433. Find The Original Array of Prefix Xor

    class Solution {
        public int[] findArray(int[] pref) {
            for(int i=pref.length-1;i>0;--i)
                pref[i]^=pref[i-1];
            return pref;
        }
    }


## 2434. Using a Robot to Print the Lexicographically Smallest String

    class Solution {
        public String robotWithString(String s) {
            int len = s.length();
            char suffixMin[] = new char[len];
            suffixMin[len-1] = s.charAt(len-1);

            for(int idx=len-2; idx>=0; idx--){
                suffixMin[idx] = suffixMin[idx+1];

                char ch = s.charAt(idx);
                if(ch<suffixMin[idx]) suffixMin[idx] = ch;
            }

            Stack<Character> stack = new Stack();
            StringBuilder str = new StringBuilder();

            for(int idx=0; idx<len; idx++){
                stack.push(s.charAt(idx));

                if(idx<len-1){
                    while(stack.size()>0 && stack.peek()<=suffixMin[idx+1]){
                        str.append(stack.pop());
                    }
                }
            }

            while(stack.size()>0) str.append(stack.pop());
            return str.toString();
        }
    }
    
    
## 2435. Paths in Matrix Whose Sum Is Divisible by K

    class Solution {
        public int numberOfPaths(int[][] grid, int k) {
            int rows = grid.length;
            int cols = grid[0].length;
            long dp[][][] = new long[rows][cols][k];
            long mod = (long)1e9+7;

            for(int row=0; row<rows; row++){
                for(int col=0; col<cols; col++){
                    int rem = grid[row][col]%k;

                    if(row==0 && col==0) dp[row][col][rem] = 1;
                    else{
                        for(int currem=0;currem<k;currem++){
                            if(row==0) dp[row][col][(currem+rem)%k] = dp[row][col-1][currem];
                            else if(col==0) dp[row][col][(currem+rem)%k] = dp[row-1][col][currem];
                            else{
                                dp[row][col][(currem+rem)%k] = ((dp[row-1][col][currem]%mod) + (dp[row][col-1][currem]%mod))%mod;
                            }
                        }
                    }
                }
            }

            int ans = (int)(dp[rows-1][cols-1][0]%mod);
            if(ans<0) ans+=(int)mod;
            return ans;
        }
    }
    
    
## 2437. Number of Valid Clock Times

    class Solution {
        public int countTime(String time) {
            char h1 = time.charAt(0);
            char h2 = time.charAt(1);
            char m1 = time.charAt(3);
            char m2 = time.charAt(4);
            int valid=1;
            if(h1=='?' && h2=='?')
                valid=valid*24;
            else if(h1!='?' && h2=='?'){
                if(h1=='0' || h1=='1')
                    valid=valid*10;
                else
                    valid=valid*4;
            }
            else if(h1=='?' && h2!='?'){
                if(Character.getNumericValue(h2)<=3)
                    valid=valid*3;
                else
                    valid=valid*2;
            }
            if(m1=='?' && m2=='?')
                valid=valid*60;
            else if(m1!='?' && m2=='?')
                valid=valid*10;
            else if(m1=='?' && m2!='?')
                valid=valid*6;
            return valid;
        }
    }
    

## 2438. Range Product Queries of Powers

    class Solution {    
        public int[] productQueries(int n, int[][] queries) {
            List<Integer> list = new ArrayList<>();
            int sum = n;
            int value = 0;
            while (sum > 0) {
                value = findPreviousPower(sum);
                sum = sum - value;
                list.add(value);
            }
            Collections.reverse(list);
            int[] result = new int[queries.length];
            int index = 0;
            for (int[] query : queries) {
                int start = query[0];
                int end = query[1];
                long queryResult = 1;
                for (int i = start; i <= end; i++) {
                    queryResult = (queryResult * list.get(i) % (long) (Math.pow(10, 9) + 7));
                }
                result[index] =  (int) queryResult;
                index++;
            }
            return result;
        }
        public int findPreviousPower(int value) {
            while ((value & value - 1) != 0) {
                value = value & value - 1;
            }
            return value;
        }
    }
    

## 2439. Minimize Maximum of Array

    class Solution:
        def minimizeArrayValue(self, nums: List[int]) -> int:
            def ispossible(target,arr):
                for i in range(len(arr)-1):
                    if arr[i] > target:
                        return False
                    arr[i+1] -= target-arr[i]
                if arr[len(arr)-1]<=target:
                    return True
                return False
            ans = 0 
            l = 0
            r = max(nums)
            while l<=r:
                mid = (l+r)>>1
                arr = nums[:]
                if ispossible(mid,arr):
                    r = mid-1
                    ans = mid
                else:
                    l = mid+1
            return ans   
            
            
## 2440. Create Components With Same Value

    class Solution {
        int[] nums;
        public int componentValue(int[] nums, int[][] edges) {
            int n = nums.length;
            this.nums = nums;
            List<Integer>[] graph = new ArrayList[n];
            for(int i=0; i<n; i++) {
                graph[i] = new ArrayList<>();
            }
            for(int[] e : edges) {
                graph[e[0]].add(e[1]);
                graph[e[1]].add(e[0]);
            }

            int sum = 0;
            for(int i : nums) {
                sum += i;
            }

            for(int k=n; k>0; k--) {
                if(sum % k != 0) continue;
                int ans = helper(graph, 0, -1, sum / k);
                if(ans == 0) return k-1;
            }
            return 0;
        }

        private int helper(List<Integer>[] graph, int i, int prev, int target) {
            if(graph[i].size() == 1 && graph[i].get(0) == prev) {
                if(nums[i] > target) return -1;
                if(nums[i] == target) return 0;
                return nums[i];
            }

            int sum = nums[i];
            for(int k : graph[i]) {
                if(k == prev) continue;
                int ans = helper(graph, k, i, target);
                if(ans == -1) return -1;
                sum += ans;
            }

            if(sum > target) return -1;
            if(sum == target) return 0;
            return sum;
        }
    }
    
    
## 2442. Count Number of Distinct Integers After Reverse Operations

    class Solution {
        public int countDistinctIntegers(int[] nums) {
            // palindromes wont chnage
            // just put originals and reversed into a set 

            int ans = 0;

            HashSet<Integer> revved = new HashSet<>();

            for(int b: nums){ 
                revved.add(b);
                revved.add(checkpali(b));
            }

            return revved.size();
        }
        private int checkpali(int n){
            int rev = 0;
            int ori = n;
            while(n > 0){

                int rem = n%10;

                rev = rev*10 + rem;

                n/=10;
            }

            return rev;
        }
    }
    
    
## 2443. Sum of Number and Its Reverse    

    class Solution {
        public boolean sumOfNumberAndReverse(int num) {
            if(num == 0)
                return true;
            for(int i = num; i >= 0; i--){
                int first = i;
                int second = num - i;
                if(possible(first, second, num))
                    return true;
            }
            return false;
        }

        private boolean possible(int first, int second, int target){
            if(first + second != target)
                return false;

            String f = String.valueOf(first);
            String s = String.valueOf(second);

            int i = f.length() - 1, j = 0;
            while(i >= 0 && f.charAt(i) == '0')
                i--;

            while(i >= 0 && j < s.length()){
                if(f.charAt(i) != s.charAt(j))
                    return false;
                i--;
                j++;
            }

            if(i == -1 && j == s.length())
                return true;
            return false;
        }
    }
    
    
## 2190. Most Frequent Number Following Key In an Array

    class Solution:
        def mostFrequent(self, nums: List[int], key: int) -> int:
            outcome = []
            for elem in range(len(nums)-1):
                if nums[elem] == key:
                    outcome.append(nums[elem+1])
            return max(Counter(outcome), key=Counter(outcome).get)
            
            
## 2206. Divide Array Into Equal Pairs

    class Solution {
       public boolean divideArray(int[] nums) {
            int[] freq = new int[501];
            for(int i = 0;i<nums.length;i++)
                freq[nums[i]]++;
            for(int i = 0;i<freq.length;i++)
                if(freq[i]%2 != 0)
                    return false;
            return true;
        }
    }
    
    
## 2210. Count Hills and Valleys in an Array

    class Solution {
        public int countHillValley(int[] nums) {
            int prev = 0; 
            int curr = 1;
            int count = 0;
            while(curr<nums.length-1){
                int next = curr+1;
                while(next<nums.length-1 && nums[curr]==nums[next]){
                    next++;
                }
                if(nums[curr] == nums[prev] || nums[prev] == nums[curr]){

                }else if(nums[curr]>nums[next] && nums[curr]>nums[prev] || nums[curr]<nums[next] && nums[curr]<nums[prev]){
                    count++;
                }
                prev = curr;
                curr++;
            }
            return count;
        }
    }
    
    
## 2239. Find Closest Number to Zero

    class Solution {
        public int findClosestNumber(int[] nums) {
            int min = -100000;
            int max = 100001;
            for(int num : nums) {
                if(num < 0) min = Math.max(min, num);
                else max = Math.min(max, num);
            }
            if(-min == max) return max;
            if(-min < max) return min;
            else return max;
        }
    }
    
    
## 2293. Min Max Game    

    class Solution {
        public int minMaxGame(int[] nums) {
            if (nums.length==1) return nums[0];
            int position=0;
            int[] a = new int[nums.length/2];
            int b=0;
            for(int i=0;i<nums.length-1;i=i+2) {
                if (position%2==0) {
                    a[b]=Math.min(nums[i], nums[i+1]);
                } else {
                    a[b]=Math.max(nums[i], nums[i+1]);
                }
                b++;
                position++;
            }
            return minMaxGame(a);
        }
    }
    
    
## 2303. Calculate Amount Paid in Taxes

    class Solution {
        public double calculateTax(int[][] brackets, int income) {
            if(income == 0) return 0;                                                        // you can remove this line

            double sum = 0, prev = 0;

            for(int i = 0; i < brackets.length; i++) {
                double salary = Math.min(brackets[i][0], income), tax = brackets[i][1];

                sum += (salary - prev) * tax;
                prev = salary;
            }

            return sum / 100;
        }
    }
    
    
## 2319. Check if Matrix Is X-Matrix

    class Solution {
        public boolean checkXMatrix(int[][] g) {
            for(int i=0;i<g.length;i++){
                for(int j=0;j<g[i].length;j++){
                    if(i == j || i + j == g.length - 1){    // This is Diagonal.
                        if(g[i][j] == 0)   // Diagonal element must be non-zero
                            return false;
                    }
                    else if(g[i][j] != 0) //Non-Diagonal elememt must be zero.
                        return false;

                }
            }
            return true;
        }
    }
    
    
## 2341. Maximum Number of Pairs in Array

    class Solution {
        public int[] numberOfPairs(int[] nums) {
            Arrays.sort(nums);
            Stack<Integer>s1=new Stack<Integer>();
            for(int i=0;i<nums.length;i++)
            {
                if(s1.contains(nums[i]))
                {
                    s1.pop();
                }
                else
                {
                    s1.push(nums[i]);
                }
            }
            int a[]=new int[2];
            a[0]=(nums.length-s1.size())/2;
            a[1]=s1.size();
            return a;
        }
    }
    
    
## 2347. Best Poker Hand

    class Solution {
        public String bestHand(int[] ranks, char[] suits) {
            int arr[]=new int[4];
            int brr[]=new int[13];
            for(char ch:suits){
                arr[ch-'a']++;
            }
            for(int i:ranks){
                brr[i-1]++;
            }
            if(5==arr[suits[0]-'a'])return "Flush";
            boolean flag=false;
            for(int i:ranks){
                if(brr[i-1]>=3)return "Three of a Kind";
                if(brr[i-1]>=2)flag=true;
            }
            if(flag)return "Pair";

            return "High Card";
        }
    }


## 219. Contains Duplicate II

    class Solution {
        public boolean containsNearbyDuplicate(int[] nums, int k) {
            Map<Integer, Integer> mp = new TreeMap<>();
            for(int i = 0; i < nums.length; ++i){
                if(mp.containsKey(nums[i]) && i  -  mp.get(nums[i]) <= k) return true;
                mp.put(nums[i], i);
            }
            return false;
        }
    }
    
    
## 2441. Largest Positive Integer That Exists With Its Negative

    class Solution {
        public int findMaxK(int[] nums) {
            Arrays.sort(nums);
            for (int i = 0; i < nums.length; i++) {
                for (int j = nums.length-1; j>=0; j--) {
                    if(nums[i]+nums[j]==0)return nums[j];
                }
            }
            return -1;
        }
    }


## 2432. The Employee That Worked on the Longest Task

    class Solution:
        def hardestWorker(self, n: int, logs: List[List[int]]) -> int:
            """ O(N)TS """
            it = ((b - a, -y) for (x, a), (y, b) in itertools.pairwise([(None, 0)] + logs))
            return -max(it)[1]
