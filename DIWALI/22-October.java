

## 12. Integer to Roman

    class Solution {
        private static final StringBuilder sb = new StringBuilder(15);

        private static final String[] thou = {"", "M", "MM", "MMM"};
        private static final String[] hund = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC","DCCC","CM"};
        private static final String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX","LXX","LXXX","XC"};
        private static final String[] ones = {"", "I", "II", "III", "IV", "V", "VI","VII","VIII","IX"};

        public String intToRoman(int num) {
            sb.setLength(0);

            sb.append(thou[num/1000]);
            sb.append(hund[(num%1000)/100]);
            sb.append(tens[(num%100)/10]);
            sb.append(ones[num%10]);

            return sb.toString();
        }
    }


## 237. Delete Node in a Linked List

    class Solution {
        public void deleteNode(ListNode node) {
            node.val=node.next.val;
            node.next=node.next.next;
        }
    }


## 2095. Delete the Middle Node of a Linked List

    class Solution {
        public ListNode deleteMiddle(ListNode head) {

            //Code here....
            if(head == null || head.next == null)
                return null;    //Checking corner case

            ListNode curr = head;
            curr = middle(curr);    //Finding the element previous to the middle node
            curr.next = curr.next.next; //Removing the middle node

            return head;

        }

        public ListNode middle(ListNode head)
        {
            ListNode slow = head, fast = head,prev = null;
            while(fast != null && fast.next != null)
            {
                prev = slow;
                slow = slow.next;
                fast = fast.next.next;
            }

            return prev;
        }
    }


## 1531. String Compression II

    class Solution {
        private int[][] dp;
        private char[] chars;
        private int n;

        public int getLengthOfOptimalCompression(String s, int k) {
            this.chars = s.toCharArray();
            this.n = s.length();
            this.dp = new int[n][k+1];
            for (int[] row: dp) {
                Arrays.fill(row, -1);
            }
            return dp(0, k);
        }

        private int dp(int i, int k) {
            if (k < 0) return n;
            if (n <= i + k) return 0;

            int ans = dp[i][k];
            if (ans != -1) return ans; 
            ans = dp(i + 1, k - 1);
            int length = 0, same = 0, diff = 0;

            for (int j=i; j < n && diff <= k; j++) {

                if (chars[j] == chars[i]) {
                    same++;
                    if (same <= 2 || same == 10 || same == 100) length++;
                } else {
                    diff++; 
                }
                ans = Math.min(ans, length + dp(j + 1, k - diff)); 
            }
            dp[i][k] = ans;
            return ans;
        }
    }


## 1335. Minimum Difficulty of a Job Schedule

    class Solution {
        public int minDifficulty(int[] jobDifficulty, int d) {
            int len = jobDifficulty.length;
            if (d > len) return -1;
            int[][] minDifficulty = new int[d][len];
            for (int i = 1; i < d; i++) {
                Arrays.fill(minDifficulty[i], Integer.MAX_VALUE);
            }
            int maxDifficulty = 0;
            for (int i = 0; i <= len - d; i++) {
                maxDifficulty = Math.max(maxDifficulty, jobDifficulty[i]);
                minDifficulty[0][i] = maxDifficulty;
            }
            for (int day = 1; day < d; day++) {
                for (int to = day; to <= len - d + day; to++) {
                    int currentDayDifficulty = jobDifficulty[to];
                    int result = Integer.MAX_VALUE;
                    for (int j = to - 1; j >= day - 1; j--) {
                        result = Math.min(result, minDifficulty[day - 1][j] + currentDayDifficulty);
                        currentDayDifficulty = Math.max(currentDayDifficulty, jobDifficulty[j]);
                    }
                    minDifficulty[day][to] = result;
                }   
            }
            return minDifficulty[d - 1][len - 1];
        }
    }



## 1832. Check if the Sentence Is Pangram
    class Solution {
        public boolean checkIfPangram(String sentence) {
            if(sentence.length()<26) return false;
             HashSet<Character> set = new HashSet<>();
            for(char c:sentence.toCharArray()){
                set.add(c);
            }

            return set.size()==26;

        }
    }


## 38. Count and Say

    class Solution {
        public String countAndSay(int n) {
            String s = "1";
            for(int i=2;i<=n;i++){
                s = countIndex(s);
            }
            return s;
        }

        public String countIndex(String s){
            StringBuilder sb = new StringBuilder();
            char c = s.charAt(0);
            int count = 1;
            for(int i=1;i<s.length();i++){
                if(s.charAt(i) == c)
                    count++;
                else{
                    sb.append(count);
                    sb.append(c);
                    c = s.charAt(i);
                    count=1;
                }
            }
            sb.append(count);
            sb.append(c);
            return sb.toString();
        }
    }



## 692. Top K Frequent Words

    class Solution {
        public List<String> topKFrequent(String[] words, int k) {
                TreeMap<String, Integer> map = new TreeMap<>(String::compareTo);
                Arrays.stream(words).forEach(x -> map.put(x, map.getOrDefault(x, 0) + 1));
                return map.entrySet().stream()
                    .sorted((o1, o2) -> Integer.compare(o2.getValue(), o1.getValue()))
                    .map(Map.Entry::getKey)
                    .limit(k)
                    .collect(Collectors.toList());
            }
    }



## 12. Integer to Roman

    class Solution {
        private static final StringBuilder sb = new StringBuilder(15);

        private static final String[] thou = {"", "M", "MM", "MMM"};
        private static final String[] hund = {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC","DCCC","CM"};
        private static final String[] tens = {"", "X", "XX", "XXX", "XL", "L", "LX","LXX","LXXX","XC"};
        private static final String[] ones = {"", "I", "II", "III", "IV", "V", "VI","VII","VIII","IX"};

        public String intToRoman(int num) {
            sb.setLength(0);

            sb.append(thou[num/1000]);
            sb.append(hund[(num%1000)/100]);
            sb.append(tens[(num%100)/10]);
            sb.append(ones[num%10]);

            return sb.toString();
        }
    }  





## 2289. Steps to Make Array Non-decreasing

    class Solution {
        public int totalSteps(int[] nums) {
            int max=0;
            Stack<Integer> st = new Stack<>();
            int[] dp = new int[nums.length];
            st.push(nums.length-1);
            for(int i=nums.length-2;i>=0;i--){
                if(nums[st.peek()]>=nums[i]){
                    st.push(i);
                }else{
                    while(!st.isEmpty() && nums[st.peek()]<nums[i]){
                        dp[i] = Math.max(dp[i]+1,dp[st.peek()]);
                        st.pop();
                    }
                    max = Math.max(max,dp[i]);
                    st.push(i);
                }
            }
            return max;
        }
    }





## 2280. Minimum Lines to Represent a Line Chart

    class Solution {
        public int maximumTop(int[] nums, int k) {
            if(k % 2 == 1 && nums.length == 1) return -1;
            int ans = -1;
            if(k > nums.length){
                for(int num : nums) ans = Math.max(ans, num);
            }
            else{
                for(int i = 0; i < k - 1; i++){
                    ans = Math.max(ans, nums[i]);
                }
                if(k < nums.length) ans = Math.max(ans, nums[k]);
            }
            return ans;
        }
    }




## 2280. Minimum Lines to Represent a Line Chart

    class Solution {
        // check for collinearity of 3 points is there slope is equal
        // => (y2 - y1) / (x2 - x1) = (y1 - y0) / (x1 - x0)
    //     => (y2 - y1) * (x1 - x0) = (y1 - y0) * (x2 - x1) (cross multiplication)
        // if lhs == rhs then points are collinear else points are not collinear
        public int minimumLines(int[][] stockPrices) {
            if(stockPrices.length == 1) return 0;

            Arrays.sort(stockPrices, (a, b) -> a[0] - b[0]);
            int lines = 1;


            for(int i = 2; i < stockPrices.length; i++){
                int x0 = stockPrices[i][0];
                int x1 = stockPrices[i - 1][0];
                int x2 = stockPrices[i - 2][0];
                int y0 = stockPrices[i][1];
                int y1 = stockPrices[i - 1][1];
                int y2 = stockPrices[i - 2][1];

                int lhs = (y2 - y1) * (x1 - x0);
                int rhs = (y1 - y0) * (x2 - x1);
                if(lhs != rhs){ // points are not collinear  -> they will not lie in same line
                    lines++;
                }
            }
            return lines;
        }
    }




## 1191. K-Concatenation Maximum Sum

    class Solution {
        public int kConcatenationMaxSum(int[] arr, int k) {
            int n = arr.length;
            if(k==1){
                long ans=0L, sum=0L;
                for(int i=0;i<n;i++){
                    sum+=arr[i];
                    ans=Math.max(ans,sum);
                    if(sum<0) sum=0;
                }
                return (int)(ans%1000000007);
            }else{
                ArrayList<Integer> ar = new ArrayList<>();
                for(int i=0;i<n;i++) ar.add(arr[i]);
                long s=0L;
                for(int i=0;i<n;i++) s+=ar.get(i);
                if(s<=0){
                    for(int i=0;i<n;i++) ar.add(ar.get(i));
                    long ans=0L, sum=0L;
                    for(int i=0;i<(n*2);i++){
                        sum+=ar.get(i);
                        ans=Math.max(ans,sum);
                        if(sum<0) sum=0;
                    }
                    return (int)(ans%1000000007);
                }else{
                    for(int i=0;i<n;i++) ar.add(ar.get(i));
                    long ans=0L, sum=0L;
                    for(int i=0;i<(n*2);i++){
                        sum+=ar.get(i);
                        ans=Math.max(ans,sum);
                        if(sum<0) sum=0;
                    }
                    for(int i=0;i<k-2;i++) ans+=s;
                    return (int)(ans%1000000007);
                }
            }
        }
    }





## 665. Non-decreasing Array

	    class Solution {
		public boolean checkPossibility(int[] nums) {
		    return checkInForwardDirection(nums) || checkInBackwardDirection(nums);
		}
		private boolean checkInForwardDirection(int[] nums){
		    int issueIndex = -1;
		    int num = 0;
		    for(int i = 0; i < nums.length - 1; i++){
			//fix the issue, if found
			if(nums[i] > nums[i + 1]){
			    issueIndex = i;
			    num = nums[i + 1];
			    nums[i + 1] = nums[i]; 
			    break;
			}
		    }
		    //no issue found, it mean nums is already non - decreasing
		    if(issueIndex == -1) return true;
		    //check once again, it becomes completely non - decreasing or not
		    //after fixing the issue
		    if(isNonDecreasing(nums)) return true;
		    //restore array back to original
		    nums[issueIndex + 1] = num;
		    return false;
		}
		private boolean checkInBackwardDirection(int[] nums){
		    for(int i = nums.length - 1; i > 0; i--){
			//fix the issue, if found
			if(nums[i] < nums[i - 1]){
			    nums[i - 1] = nums[i];
			    break;
			}
		    }
		    //check once again, it becomes completely non - decreasing or not
		    return isNonDecreasing(nums); 
		}
		private boolean isNonDecreasing(int[] nums){
		    for(int i = 0; i < nums.length - 1; i++){
			if(nums[i] > nums[i + 1]) return false;
		    }
		    return true;
		}
	    }



## 2195. Append K Integers With Minimal Sum

	class Solution {
	    public long minimalKSum(int[] nums, int k) {
		TreeSet<Integer> set = new TreeSet<Integer>();
		for (int n : nums)                               
		    set.add(n);
		long sum = ((long)(k + 1) * (long)k) / 2;     
		int count = 0;                                
		for (Integer i : set) {
		    if (i > k)                      
			break;
		    else {
			sum -= i;                              
			count++;                              
		    }                                          
		}
		int i = k + 1;                              
		while (count > 0) {
		    if (!set.contains(i)) {
			sum += i;
			count--;
		    }
		    i++;
		}
		return sum;
	    }
	}



## 2333. Minimum Sum of Squared Difference

	class Solution {
	    public long minSumSquareDiff(int[] nums1, int[] nums2, int k1, int k2) {
		int lo=0,hi=100000,k=k1+k2;
		long extra=0,ans=0;
		while(lo<hi){
		    int mid=(lo+hi)>>1;
		    long need=0;
		    for (int i = 0 ; i< nums1.length&&need<=k;i++){
			need += Math.max(0, Math.abs(nums1[i]-nums2[i])-mid);
		    }
		    if (need<=k){
			hi=mid;
			extra=k-need;
		    }else{
			lo=mid+1;
		    }
		}

		for (int i = 0; i< nums1.length&&lo>0;i++){// make sure to check lo (diff) > 0 here.
		    long diff = Math.min(lo, Math.abs(nums1[i]-nums2[i]));
		    if (diff==lo&&extra>0){ 
			--diff;
			--extra;
		    }
		    ans+=diff*diff;
		}

		return ans;
	    }
	}




## 805. Split Array With Same Average

	class Solution 
	{
	    public boolean splitArraySameAverage(int[] nums)
	    {
		if(nums.length<=1)
		    return false;
		int n = nums.length;
		//finding sum of elements of the array :- nums
		int sum = 0;
		for(int i=0 ; i<n ; i++)
		{
		    sum += nums[i];
		}
		// We divide the given array into 2 parts , where n1 = length of part-1 & s1 = sum of elements of part-1
		// s1 = (sum*n1)/n , since the average of both part-1 & part-2 are equal
		// Range of n1 = [ 1 , n-1 ]
		// We iterate over each possible valid value of n1 , and check if the corresponding subseq of sum = s1 & length = n1 is possible or not
		for(int n1=1 ; n1<=(n-1) ; n1++)
		{
		    int numerator = (sum*n1);
		    int denominator = n;
		    int s1 = numerator/denominator;
		    // if the s1 = float value , then we move on to the next possible n1
		    if(numerator%denominator !=0) 
			continue;
		    //if s1 = integer , then we check if there exists a subseq whose sum = s1 , we check this by dp
		    Boolean dp[][] = new Boolean[n+1][s1+1];
		    boolean isTargetPresent = isPresent( s1 ,0,0,n1, nums, nums.length , dp);
		    if(isTargetPresent)
			return true;
		}
		return false;
	    }
		// function that check if there exists a subseq. of length=n1 & sum=s1
	   private boolean isPresent(int target,int count,int sum,int n1, int[] nums, int n , Boolean dp[][])
	   {
	       if(n==0)
		{
		    if(target==0 && count==n1)
			return true;
		    else
			return false;
		}
		if(count==n1)
		{
		    if(target==0)
			return true;
		    else
			return false;
		}
		if(dp[n][target] != null)
		    return dp[n][target];
		if(nums[n-1] > target)
		{
		    boolean skip = isPresent(target,count,sum,n1,nums,n-1,dp);
		    dp[n][target] = skip;
		    return dp[n][target];
		}
		else
		{
		    boolean include = isPresent(target-nums[n-1] , count+1 , sum + nums[n-1] , n1 , nums , n-1 , dp);
		    boolean exclude = isPresent(target , count , sum , n1 , nums , n-1 , dp );
		    dp[n][target] = include || exclude;
		    return dp[n][target];
		}
	\
	   }
	}



## 47. Permutations II

	class Solution {
	    List<List<Integer>> ml;
	    public List<List<Integer>> permuteUnique(int[] nums) {
		//main list to store permutations.
		ml = new ArrayList<>();
		//visited array to mark an element when it is visited.
		boolean[] visited = new boolean[nums.length];
		//Sort the elements.
		Arrays.sort(nums);

		findPermutations(nums, new ArrayList<>(), visited);

		return ml;
	    }

	    public void findPermutations(int[] nums, List<Integer> cl, boolean[] visited) {
		//When size of child list is equal to length of nums array, meaning a permutation is formed, 
			//so add it in main list.
		if (cl.size() == nums.length) {
		    ml.add(new ArrayList<>(cl));
		    return;
		}

		for (int i = 0; i < nums.length; i++) {
		    //If the element is visited or it is a duplicate, then continue.
		    if (visited[i] || (i > 0 && nums[i] == nums[i - 1]) && !visited[i - 1]) {
			continue;
		    }

		    //Mark the element visited.
		    visited[i] = true;
		    //Add it in child list.
		    cl.add(nums[i]);

		    findPermutations(nums, cl, visited);

		    //Backtracking : Remove the recently added element.
		    cl.remove(cl.size() - 1);
		    //Mark the element unvisited.
		    visited[i] = false;
		}
	    }
	}



## 31. Next Permutation

	class Solution {
	    public void nextPermutation(int[] arr) {
		int n = arr.length;

		int pivot = -1;
		for(int i=n-2; i>=0; --i){
		    if(arr[i]<arr[i+1]){
			pivot=i; 
			break;
		    }
		}
		// this means no valid index found to generate next higher permutation.
		if(pivot==-1){
		    reverseArr(arr, -1); // reversing whole arr
		    return;
		}

		int biggerInd= -1;
		for(int i=n-1; i>pivot; --i){
		    if(arr[i] > arr[pivot]){  // after pivot getting first greater value.
			biggerInd = i;
			break;
		    }
		}

		swap(arr, pivot, biggerInd); // swapping bigger val to its appropriate place.
		//since, pivot was the 1st smaller index, this means before pivot values must be greater & reversing is required to get the next higher permutation.
		reverseArr(arr, pivot);   

	    }

	    public void swap(int[] arr, int i, int j){
		int temp = arr[i];
		arr[i] = arr[j];
		arr[j] = temp;
	    }

	    public void reverseArr(int[] arr, int ind){
		int l=ind+1;
		int r=arr.length-1;
		while(l<=r){
		    swap(arr, l, r);
		    l++; r--;
		}
	    }

	}



## 37. Sudoku Solver

	class Solution {
	    private boolean isSafe(char[][] board,int i,int j,int c){
		for (int p = 0; p < 9; p++) {
		    if(board[p][j]==c)
			return false;
		}
		for (int p = 0; p < 9; p++) {
		    if(board[i][p]==c)
			return false;
		}

		int row = i-i%3;
		int col = j-j%3;
		for(int p = row;p<row+3;p++){
		    for(int q = col;q<col+3;q++){
			if(board[p][q] == c){
			    return false;
			}
		    }
		}
		return true;
	    }
	    public boolean dfs(char[][] board){
		for(int i=0;i<9;i++){
		    for(int j=0;j<9;j++){
			if(board[i][j]=='.'){
			    for(char k='1';k<='9';k++){
				if(isSafe(board,i,j,k)){
				    board[i][j] = k;
				    if(dfs(board))
					return true;
				    else
					board[i][j]='.';
				}

			    }
			    // this is bcoz we are unable to generate a valid sudoko with all the possible explorations.
			    return false;
			}
		    }
		}
		return true;
	    }
	    public void solveSudoku(char[][] board) {
		if(board.length==0) {
		    return;
		}
		dfs(board);
	    }
	}



## 40. Combination Sum II

	class Solution {
	    List<List<Integer>> ans = new ArrayList<>();
	    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		Arrays.sort(candidates);
		backtracking(candidates,target,0,new ArrayList<>());
		return ans;
	    }
	    public void backtracking(int [] candidates,int target,int rejected, ArrayList<Integer> list){
		if(target<0){
		    return;
		}
		if(target==0){
		    ans.add(new ArrayList<>(list));
		    return;
		}
		for(int i=rejected;i<candidates.length;i++){
		    if(i>rejected && candidates[i]==candidates[i-1]){
			continue;
		    }
		    list.add(candidates[i]);
		    backtracking(candidates,target-candidates[i],i+1,list);
		    list.remove(list.size()-1);
		}
	    }
	}






## 295. Find Median from Data Stream

	class MedianFinder {

	    private Queue<Integer> largeNumMinHeap;                     // Min-Heap storing larger numbers, will store at most k ints
	    private Queue<Integer> smallNumsMaxHeap;                    // Max-Heap storing smaller numbers, will store at most k + 1 ints

	    public MedianFinder() {
		this.largeNumMinHeap = new PriorityQueue<Integer>((a, b) -> Integer.compare(a, b));
		this.smallNumsMaxHeap = new PriorityQueue<Integer>((a,b) -> Integer.compare(b, a));
	    }
	    public void addNum(int num) {
		smallNumsMaxHeap.offer(num);
		largeNumMinHeap.offer(smallNumsMaxHeap.poll());
		if (smallNumsMaxHeap.size() < largeNumMinHeap.size())
		{
		    smallNumsMaxHeap.offer(largeNumMinHeap.poll());
		}
	    }
	    public double findMedian() {
		if (smallNumsMaxHeap.size() > largeNumMinHeap.size())
		{
		    return smallNumsMaxHeap.peek();
		}
		else
		{
		    return (smallNumsMaxHeap.peek() + largeNumMinHeap.peek()) / 2.0;
		}
	    }
	}


239. Sliding Window Maximum

	class Solution {
	    public int[] maxSlidingWindow(int[] nums, int k) {
		Deque<Integer> dq = new LinkedList();
		int n = nums.length;
		int ans[] = new int[n-k+1];

		for(int l = 0, r = 0; r < n; r++)
		{
		    // Left boundary of the window.
		    l = r - k + 1;
		    int curr = nums[r];

		    // Check recently added index and remove it if less than current element,
		    // to maintain the decreasing order in the Queue(deck).
		    while(!dq.isEmpty() && nums[dq.peekLast()] < curr)
			dq.pollLast();
		    dq.addLast(r);

		    // Remove the left index from queue, if it is outside the window.
		    if(!dq.isEmpty() && l > dq.peekFirst())
			dq.pollFirst();

		    if(r+1 >= k)
		    {
			ans[l] = nums[dq.peekFirst()];
			l++;
		    }
		}

		return ans;
	    }
	}


## 124. Binary Tree Maximum Path Sum

	class Solution {

	    public int maxPathSum(TreeNode root) {

		 int[] maxSum = new int[1];
		 maxSum[0] = Integer.MIN_VALUE; 

		findPathSum(root,maxSum);
		return maxSum[0];
	    }
	    private int findPathSum(TreeNode root,int[] maxSum) {
		if(root == null) return 0;

		int left = Math.max(0 , findPathSum(root.left , maxSum));
		int right = Math.max(0 , findPathSum(root.right , maxSum));

		maxSum[0] = Math.max(maxSum[0] , left + right + root.val);

		return root.val + Math.max(left , right);
	    }
	}



## 51. N-Queens

	class Solution {
	  public List<List<String>> solveNQueens(int n) {
	    boolean[][] board = new boolean[n][n];
	    List<List<String>> answer = new ArrayList<>();
	    queens(board,0,answer);
	    return answer;
	}
	void queens(boolean[][] board,int row,List<List<String>> answer){

	    if(row==board.length){
		List<String> res = insert(board);
		answer.add(res);
		return;
	    }

	    // Placing Queens and Checking for every row and column
	    for(int col=0;col<board.length;col++){
		//place queen if Safe
		if(isSafe(board,row,col)){
		    board[row][col] = true;
		    queens(board,row+1,answer); //Recursive call
		    board[row][col] = false;    // Backtrack
		}
	    }
	}

	boolean isSafe(boolean[][] board, int row ,int col){

	    // Vertical row
	    for(int i=0;i<row;i++){
		if(board[i][col])
		    return false;
	    }

	    // Left diagonal
	    int maxLeft = Math.min(row,col);
	    for(int i=1;i<=maxLeft;i++){
		if(board[row-i][col-i]){
		    return false;
		}
	    }

	    // Right diagonal
	    int maxRight = Math.min(row,board.length-col-1);
	    for(int i=1;i<=maxRight;i++){
		if(board[row-i][col+i]){
		    return false;
		}
	    }
	    return true;
	}

	  List<String> insert(boolean[][] board){

	    List<String> ans = new ArrayList<>();
	    for(boolean[] row: board){
		String S = "";
		for(boolean element:row){
		    if(element){
			S += "Q";
		    }
		    else
			S += ".";
		}
		ans.add(S);
	    }
	      return ans;
	}
	}



## 32. Longest Valid Parentheses

	class Solution {
	public int longestValidParentheses(String s) {
		if(s==null){
		    return 0;
		}
		Stack<Integer> stk = new Stack();
		int result = 0;
		int lastI = -1; //it will remain -1 for ()() , (()). Will change if extra closing ')' is encountered e.g. ) ()()
	    // note we are pushing only ( in the stack
		for(int i=0; i<s.length(); i++) {
		    if(s.charAt(i)=='(') {
			stk.push(i);      
		    } 

		    else {

			if(!stk.isEmpty()) { // there is matching '(' in stack
			    stk.pop();  //remove matching '('
			    //find valid string start based on stack emptiness

			    if(!stk.isEmpty()) { 
				//there are unclosed '(' other than the one popped '(' in prev line e.g. for (()  , ( is still in stack
				result = Math.max(result, i-stk.peek());
				//so valid str len will start after unclosed '(' which is still in stack 
			    }
			    else 
			    {
				result = Math.max(result, i-lastI); //stack is empty now, use lastI to find the begining which is either after 0 for which lastI = -1 for it or  after extra ) e.g. ()() -> lastI =-1 or )()() -> lastI = 0
			    }    

			} 

			else //(while stack is empty and we get )
			{ 
			    // there is extra ) that has no matching '(', preserve the lastI
			    // this is case for like )()()
			    lastI = i;
			}
		    }
		}
		return result;
	    }
	 } 



## 25. Reverse Nodes in k-Group

	class Solution {
	    public ListNode reverseKGroup(ListNode head, int k) {
	       if(k==1){
		   return head;
	       } 
		ListNode left=head;
		ListNode right=head;
		ListNode prev=null;
		int j=0;
		while(right!=null && right.next!=null){
		int i=1;
		    while(i<k &&right!=null){
		     right=right.next; 
		    i++;
		   }
		   if(right!=null){
		       reverse(left,right,prev);
		       j++;
		   }else{
		       break;
		   }
		    if(j==1){
			head=right;
		    }
		    if(prev!=null){
		   prev.next=right;
		    }
		   prev=left;
		   right=left.next;
		   left=left.next;
		} 
		return head;
	    }
	    public void reverse(ListNode left,ListNode right,ListNode prev){
		ListNode temp=left;
		ListNode last=prev;
	      while(prev!=right){

	       ListNode frwd=left.next;
		left.next=prev;
		prev=left;
		left=frwd;
	     }
	      temp.next=left;
		if(last!=null){
		last.next=right;
		} 
	    }
	}



## 23. Merge k Sorted Lists

	class Solution {
	    public ListNode mergeKLists(ListNode[] lists) {

	/*
	Intuition behind the approach is to initialise a min heap and then add every element of the list to the min heap.
	Since it is a min heap, therefore root will be the smallest element and we can extract it and add to the new list.


	*/
		PriorityQueue<Integer> pq = new PriorityQueue<Integer>();
		//ading everything to the new priority queue
		for(ListNode ln : lists) {
		    while(ln != null) {
			pq.add(ln.val);
			ln = ln.next;
		    }
		}
		//list to store the sorted list
		ListNode head = new ListNode(0);
		ListNode h = head;

		while(!pq.isEmpty()) {
		    ListNode t = new ListNode(pq.poll());
		    h.next = t;
		    h = h.next;
		}

		return head.next;

	    }
	}



