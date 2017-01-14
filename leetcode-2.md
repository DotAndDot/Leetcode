+ 123 Best Time to Buy and Sell Stock III

  动态规划：dp1数组为0 - i 天最大获利，dp2位i - (len - 1)天最大获利

  ```c++
  class Solution {
  public:
      int maxProfit(vector<int>& prices) {
          int min, max, len = prices.size();
          if(len == 0)
              return 0;
          int dp1[len] = {0}, dp2[len] = {0};
          min = prices[0];
          for(int i = 1; i < len; i++){
              if(prices[i] - min > dp1[i - 1]){
                  dp1[i] = prices[i] - min;
              }
              else
                  dp1[i] = dp1[i - 1];
              if(prices[i] < min)
                  min = prices[i];
          }
          max = prices[len - 1];
          dp2[len - 1] = 0;
          for(int i = len - 2; i >= 0; i--){
              if(max - prices[i] > dp2[i + 1]){
                  dp2[i] = max - prices[i];
              }
              else
                  dp2[i] = dp2[i + 1];
              if(prices[i] > max)
                  max = prices[i];
          }
          max = 0;
          for(int i = 0; i < len; i++){
              if(dp1[i] + dp2[i] > max)
                  max = dp1[i] + dp2[i];
          }
          return max;
      }
  };
  ```

+ 287 Find the Duplicate Number

  二分法：

  实际上，我们可以根据抽屉原理简化刚才的暴力法。我们不一定要依次选择数，然后看是否有这个数的重复数，我们可以用二分法先选取n/2，按照抽屉原理，整个数组中如果小于等于n/2的数的数量大于n/2，说明1到n/2这个区间是肯定有重复数字的。比如6个抽屉，如果有7个袜子要放到抽屉里，那肯定有一个抽屉至少两个袜子。这里抽屉就是1到n/2的每一个数，而袜子就是整个数组中小于等于n/2的那些数。这样我们就能知道下次选择的数的范围，如果1到n/2区间内肯定有重复数字，则下次在1到n/2范围内找，否则在n/2到n范围内找。下次找的时候，还是找一半。

  ```c++
  class Solution {
  public:
  int findDuplicate(vector<int>& nums) {
      int left = 0, right = nums.size() - 1, mid;
      int cnt;
      while(left <= right){
          mid = left + (right - left) / 2;
          cnt = 0;
          for(int i = 0; i < nums.size(); i++){
              if(nums[i] <= mid)
                  cnt++;
          }
          if(cnt > mid)
              right = mid - 1;
          else
              left = mid + 1;
      }
      return left;
  }
  };
  ```

  佛洛依德判圈法：
  设已知某个起点节点为节点S。现设两个指针t和h，将它们均指向S。
  接着，同时让t和h往前推进，但是二者的速度不同：t每前进1步，h前进2步。只要二者都可以前进而且没有相遇，就如此保持二者的推进。当h无法前进，即到达某个没有后继的节点时，就可以确定从S出发不会遇到环。反之当t与h再次相遇时，就可以确定从S出发一定会进入某个环，设其为环C。
  如果确定了存在某个环，就可以求此环的起点与长度。

  上述算法刚判断出存在环C时，显然t和h位于同一节点，设其为节点M。显然，仅需令h不动，而t不断推进，最终又会返回节点M，统计这一次t推进的步数，显然这就是环C的长度。

  为了求出环C的起点，只要令h仍位于节点M，而令t返回起点节点S。随后，同时让t和h往前推进，且保持二者的速度相同：t每前进1步，h前进1步。持续该过程直至t与h再一次相遇，设此次相遇时位于同一节点P，则节点P即为从节点S出发所到达的环C的第一个节点，即环C的一个起点。

  n + 1 个1至n的数，数组中能构成环（无法到达n+1坐标）；

  ```c++
  int findDuplicate(vector<int>& nums) {
      int slow = 0;
  	int fast = 0;
  	int finder = 0;

  	while (true)
  	{
  		slow = nums[slow];
  		fast = nums[nums[fast]];

  		if (slow == fast)
  			break;
  	}
  	while (true)
  	{
  		slow = nums[slow];
  		finder = nums[finder];
  		if (slow == finder)
  			return slow;
  	}
  }
  ```

+ 101 Symmetric Tree

  ```c++
  class Solution {
  public:
      bool isSymmetric(TreeNode* left, TreeNode* right){
          if(left == NULL && right == NULL)
              return true;
          else if(left == NULL || right == NULL)
              return false;
          if(left->val != right->val)
              return false;
          else
              return (isSymmetric(left -> left, right -> right) && isSymmetric(left -> right, right -> left));
      }
      bool isSymmetric(TreeNode* root) {
          if(root == NULL)
              return true;
          return isSymmetric(root -> left, root -> right);
      }
  };
  ```

  ```c++
  class Solution {
  public:
      void searchleft(TreeNode* p, stack<int>& s, int l){
          if(p == NULL)
              return;
          searchleft(p -> left, s, l + 1);
          s.push(l);
          s.push(p -> val);
          searchleft(p -> right, s, l + 1);
      }
      bool judgeright(TreeNode* p, stack<int>& s, int l){
          if(p == NULL)
              return true;
          if(!judgeright(p -> left, s, l + 1))    return false;

              if(s.size() > 0 && s.top() == p -> val){
                  s.pop();
                  if(s.size() > 0 && s.top() == l){
                      s.pop();
                  }
                  else
                      return false;
              }
              else
                  return false;
           if(!judgeright(p -> right, s, l + 1) )  return false;
           return true;
      }
      bool isSymmetric(TreeNode* root) {
          if(root == NULL)
              return true;
          if(root->left == NULL && root->right != NULL || root->left !=NULL && root->right == NULL)
              return false;
          stack<int> s;
          searchleft(root -> left, s, 1);
          cout<<endl;
          if(judgeright(root -> right, s, 1) && s.size() == 0)
              return true;
          return false;
      }
  };
  ```

+ 110 Balanced Binary Tree

  平衡二叉树 AVL 树
  1.平衡因子：二叉排序树中 Balance Fector(BF)=|左子树的深度-右子树的深度|**
  2.平衡二叉树的性质
   (1)树中节点的BF<=1
   (2)左右子树都是平衡二叉树
  ```c++
  class Solution {
    public:
        int judge(TreeNode* p){
            if(p == NULL)
                return 0;
            int ld = judge(p->left), rd = judge(p->right);
            if(abs(ld - rd) > 1 || ld == -1 || rd == -1)
                return -1;
            return max(ld, rd) + 1;
        }
        bool isBalanced(TreeNode* root) {
            if(root == NULL)
                return true;
            if(judge(root) == -1)
                return false;
            return true;
        }
  };
  ```
+ 207 Course Schedule

  拓扑排序：
  dfs:从一个未访问的节点开始依次遍历与该节点相连的节点，若遇到了在该次遍历中已遇到的节点则证明存在环，即不能拓扑排序，需要维护两个数组visited和onpath，visitied记录已经遍历过的节点，onpath记录当次遍历所经过的节点
  ```c++
  class Solution {
  public:
      bool dfstopo(vector<unordered_set<int>>& graph, int index, vector<bool>& visited, vector<bool>& onpath){
          if(visited[index])  return false;
          onpath[index] = visited[index] = true;
          for(auto a : graph[index]){
              if(onpath[a] || dfstopo(graph, a, visited, onpath))
                  return true;
          }
          return onpath[index] = false;
      }
      bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
          vector<unordered_set<int>> graph(numCourses);
          for(auto a : prerequisites){
              graph[a.first].insert(a.second);
          }
          vector<bool> visited(numCourses, false), onpath(numCourses, false);
          for(int i = 0; i < numCourses; i++){
              if(!visited[i] && dfstopo(graph, i, visited, onpath))
                  return false;
          }
          return true;
      }
  };
  ```
  bfs:用数组indegree记录每个节点的入度，每次从入度为0的节点中选出1个节点，更新indegree表，如此循环，若已无未访问的节点则存在拓扑排序，若剩余节点中没有入度为0的节点则证明不存在拓扑排序
  ```c++
  class Solution {
  public:
      bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
          vector<unordered_set<int>> graph(numCourses);
          for(auto a : prerequisites){
              graph[a.second].insert(a.first);
          }
          int indegree[numCourses] = {0};
          for(auto a : graph){
              for(auto aa : a){
                  indegree[aa]++;
              }
          }
          for(int i = 0; i < numCourses; i++){
              int j = 0;
              for(; j < numCourses; j++)
                  if(indegree[j] == 0) break;
              if(j == numCourses) return false;
              indegree[j] = -1;
              for(auto a : graph[j]){
                  indegree[a]--;
              }
          }
          return true;
      }
  };
  ```
+ 210 Course Scherule II

    dfs求拓扑排序时需要将结果倒序才是正确顺序
  ```c++
  class Solution {
  public:
      bool dfstopo(vector<int>& res, vector<unordered_set<int>>& graph, int index, vector<bool>& visited, vector<bool>& onpath){
          if(visited[index])  return true;
          onpath[index] = visited[index] = true;
          for(auto a : graph[index]){
              if(onpath[a] || !dfstopo(res, graph, a, visited, onpath)){
                  return false;
              }
          }
          res.push_back(index);
          onpath[index] = false;
          return true;
      }
      vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
          vector<unordered_set<int>> graph(numCourses);
          vector<bool> visited(numCourses, false), onpath(numCourses, false);
          vector<int> res;
          vector<int> indegree(numCourses, 0);
          for(auto a : prerequisites){
              graph[a.second].insert(a.first);
          }
          for(int i = 0; i < numCourses; i++){
              if(!visited[i] && !dfstopo(res, graph, i, visited, onpath)){
                  vector<int> empty;
                  return empty;
              }
          }
          reverse(res.begin(), res.end());
          return res;
      }
  };
  ```
+ 337 House Robber III

  深度优先遍历二叉树。拿root（第0层）为例，如果取第0层的节点，则第1层的节点不能取；如果不取第0层的节点，则第1层的节点可取可不取。这样我们需要记录下每个节点取与不取时能够获取的最大钱数，通过深度优先遍历二叉树，最后取root节点返回的两个数值中大的就可以了。
  ```c++
  class Solution {
  public:
      vector<int> dfs_max(TreeNode* p){
          vector<int> res(2, 0);
          if(p == NULL)   return res;
          vector<int> l = dfs_max(p->left);
          vector<int> r = dfs_max(p->right);
          res[0] = l[1] + r[1] + p->val;
          res[1] = max(l[0], l[1]) + max(r[0], r[1]);
          return res;
      }
      int rob(TreeNode* root) {
          if(root == NULL)
              return 0;
          vector<int> res = dfs_max(root);
          return max(res[0], res[1]);
      }
  };
  ```
+ 116 Populating Next Right Pointers in Each Node

  要求constant extra space，可以利用已经确定的父节点的next指针

+ 394 Decode String

  递归实现，返回值为一对括号内decode后的字符串

+ 98 Validate Binary Search Tree
  ```c++
  class Solution {
  public:
      bool dfsisvalid(TreeNode* p, int l, int r, int ls, int rs){
          if(p == NULL)   return true;
          if(rs == 1 && p->val >= r || ls == 1 && p->val <= l)    return false;
          return dfsisvalid(p->left, l, p->val, ls, 1) && dfsisvalid(p->right, p->val, r, 1, rs);
      }
      bool isValidBST(TreeNode* root) {
          if(root == NULL)    return true;
          return dfsisvalid(root->left, INT_MIN, root->val, 0, 1) && dfsisvalid(root->right, root->val, INT_MAX, 1, 0);
      }
  };
  ```
+ 322 Reconstruct Itinerary

    本题是关于有向图的边的遍历。每张机票都是有向图的一条边，我们需要找出一条经过所有边的路径，那么DFS不是我们的不二选择。先来看递归的结果，我们首先把图建立起来，通过邻接链表来建立。由于题目要求解法按字母顺序小的，那么我们考虑用multiset，可以自动排序。等我们图建立好了以后，从节点JFK开始遍历，只要当前节点映射的multiset里有节点，我们取出这个节点，将其在multiset里删掉，然后继续递归遍历这个节点，由于题目中限定了一定会有解，那么等图中所有的multiset中都没有节点的时候，我们把当前节点存入结果中，然后再一层层回溯回去，将当前节点都存入结果，那么最后我们结果中存的顺序和我们需要的相反的，我们最后再翻转一下即可
    ```c++
    class Solution {
    public:
        void dfs(unordered_map<string, multiset<string>>& graph, string s, vector<string>& res){
            while(graph[s].size()){
                string t = *graph[s].begin();
                graph[s].erase(graph[s].begin());
                dfs(graph, t, res);
            }
            res.push_back(s);
        }
        vector<string> findItinerary(vector<pair<string, string>> tickets) {
            unordered_map<string, multiset<string>> graph;
            for(auto a : tickets){
                graph[a.first].insert(a.second);
            }
            vector<string> res;
            dfs(graph, "JFK", res);
            return vector<string>(res.rbegin(), res.rend());
        }
    };
    ```
+ 117 Populating Next Right Pointers in Each Node II
    ```c++
    void connect2(TreeLinkNode * n) {  
        while (n) {  
            TreeLinkNode * next = NULL; // the first node of next level  
            TreeLinkNode * prev = NULL; // previous node on the same level  
            for (; n; n=n->next) {  
                if (!next) next = n->left?n->left:n->right;  

                if (n->left) {  
                    if (prev) prev->next = n->left;  
                    prev = n->left;  
                }  
                if (n->right) {  
                    if (prev) prev->next = n->right;  
                    prev = n->right;  
                }  
            }  
            n = next; // turn to next level  
        }  
    }  
    ```
+ 19 Remove Nth Node From End of List
    
    要求只遍历一次
    ```c++
    class Solution {
    public:
        ListNode* removeNthFromEnd(ListNode* head, int& n) {
            auto node = new ListNode(0);
            node->next = head;
            removeNode(node, n);
            return node->next;
        }

        void removeNode(ListNode* head, int& n){
            if(!head) return;
            removeNode(head->next, n);
            n--;
            if(n == -1) {
                head->next = head->next->next;
            }
            return;
        }
    };
    ```

+ 453 Minimum Moves to Equal Array Elements

    因为每个数都会经历递增的过程，最后达到一个ceiling。假设数组元素最终为X，数组最小元素min需要经过X-min次增长，最大元素max需要经过X-max次增长，(X-min)-(X-max)=max-min就是max不变  其余元素包括min 增长的次数，经过这些次增长后，min元素和max元素大小相等，且它俩成为当前数组最小元素。   然后我们再让这俩最小元素增长到当前数组最大元素（初始数组次最大元素max2）的大小，增长的次数是max2-min，最终使得这三个元素相等。每一次增长都让数组中大小相同的元素增加一个，从1到2到3~~~n，故总共增加了max-min,max2(初始数组次最大元素)-min,max3-min，，，总和就是sum-min*n

+ 258 Add Digits

    12345 = 1 * 9999 + 2 * 999 + 3 * 99 + 4 * 9 + 5 + (1+ 2+ 3 + 4 + 5)
    只要证明：12345 % 9 = (1 + 2 + 3 + 4 +5 ) % 9 就能往下递推了。
    那么，我们已知：
    m % 9 = a; n % 9 = b 即 m = 9 * x + a; n = 9 * y + b；可推出 (m + n) % 9 = a + b = m % 9 + n % 9；
    [1 * 9999 + 2 * 999 + 3 * 99 + 4 * 9 + (1+ 2+ 3 + 4 + 5)] % 9 = (1 * 9999) % 9 + (2 * 999) % 9 + (3 * 99) % 9 + (4 * 9) % 9 + (1+ 2+ 3 + 4 + 5) % 9 = 0 + 0 + 0 + 0 + (1 + 2 + 3 + 4 + 5) % 9 = (1 + 2 + 3 + 4 + 5) % 9。
    证明完成：12345 % 9 = (1 + 2 + 3 + 4 + 5) % 9 ;
    因为题中最后一个数恰好是小于10，与取mod 9结束也一致，所以：
    (12345) % 9 = (1 + 2 + 3 + 4 + 5) % 9 = 12 % 9 = (1 +2) % 9 = 3 % 9 = 3。
    一个数x的数根为mod(x-1,9)+1.因为正整数对9取模的结果取值为[0,8],,而数根的取值为[1,9]。

+ 172 Factorial Trailing Zeroes

    对n!做质因数分解n!=2x*3y*5z*...
    显然0的个数等于min(x,z)，并且min(x,z)==z
    证明：
    对于阶乘而言，也就是1*2*3*...*n
    [n/k]代表1~n中能被k整除的个数
    那么很显然
    [n/2] > [n/5] (左边是逢2增1，右边是逢5增1)
    [n/2^2] > [n/5^2](左边是逢4增1，右边是逢25增1)
    ……
    [n/2^p] > [n/5^p](左边是逢2^p增1，右边是逢5^p增1)
    随着幂次p的上升，出现2^p的概率会远大于出现5^p的概率。
    因此左边的加和一定大于右边的加和，也就是n!质因数分解中，2的次幂一定大于5的次幂

+ 326 Power of Three
    1. 任何一个3的x次方一定能被int型里最大的3的x次方整除
    2. 利用对数的换底公式来做，高中学过的换底公式为logab = logcb / logca，那么如果n是3的倍数，则log3n一定是整数，我们利用换底公式可以写为log3n = log10n / log103，注意这里一定要用10为底数，不能用自然数或者2为底数，否则当n=243时会出错，原因请看这个帖子。现在问题就变成了判断log10n / log103是否为整数，在c++中判断数字a是否为整数，我们可以用 a - int(a) == 0 来判断

+ 204 Count Primes

    埃拉托斯特尼筛法Sieve of Eratosthenes中，这个算法的过程如下图所示，我们从2开始遍历到根号n，先找到第一个质数2，然后将其所有的倍数全部标记出来，然后到下一个质数3，标记其所有倍数，一次类推，直到根号n，此时数组中未被标记的数字就是质数。

+ 264 Ugly Number II 

    动态规划

    这道题是之前那道Ugly Number 丑陋数的延伸，这里让我们找到第n个丑陋数，还好题目中给了很多提示，基本上相当于告诉我们解法了，根据提示中的信息，我们知道丑陋数序列可以拆分为下面3个子列表：

    (1) 1×2, 2×2, 3×2, 4×2, 5×2, …
    (2) 1×3, 2×3, 3×3, 4×3, 5×3, …
    (3) 1×5, 2×5, 3×5, 4×5, 5×5, …

    仔细观察上述三个列表，我们可以发现每个子列表都是一个丑陋数分别乘以2,3,5，而要求的丑陋数就是从已经生成的序列中取出来的，我们每次都从三个列表中取出当前最小的那个加入序列

+ 462 Minimum Moves to Equal Array Elements II

    中位数，不是平均数！！！

+ 69 Sqrt(x)

    1. 二分查找

    对于一个非负数n，它的平方根不会小于大于（n/2+1）（谢谢@linzhi-cs提醒）。在[0, n/2+1]这个范围内可以进行二分搜索，求出n的平方根。
    2. 牛顿迭代法

+ 413 Arithmetic Slices

    动态规划：创建两个数组dp1 和 dp2，dp1[i]表示i位作为等差数列最后一位时的数列数，dp2[i]表示i位不是等差数列最后一位时的数列数，两者相加即为i位所包含的等差数列的总数；
    ```c++
    class Solution {
    public:
        int numberOfArithmeticSlices(vector<int>& A) {
            if(A.size() < 3)    return 0;
            vector<int> dp1(A.size(), 0), dp2(A.size(), 0);
            if(A[2] - A[1] == A[1] - A[0])   dp1[2] = 1;
            for(int i = 3; i < A.size(); i++){
                if(A[i] - A[i - 1] == A[i - 1] - A[i - 2]){
                    dp1[i] = dp1[i - 1] + 1;
                }
                else
                    dp1[i] = 0;
                dp2[i] = dp1[i - 1] + dp2[i - 1];
            }
            return dp1.back() + dp2.back();
        }
    };
    ```
    http://www.cnblogs.com/grandyang/p/5968340.html

+ 50 Pow(x, n)
    1. 把n看成是以2为基的位构成的，因此每一位是对应x的一个幂数，然后迭代直到n到最高位。比如说第一位对应x，第二位对应x*x,第三位对应x^4,...,第k位对应x^(2^(k-1)),可以看出后面一位对应的数等于前面一位对应数的平方，所以可以进行迭代。因为迭代次数等于n的位数，所以算法的时间复杂度是O(logn)
    2. 二分法:递归关系可以表示为pow(x,n) = pow(x,n/2)*pow(x,n-n/2)

+ 21 Merge Two Sorted Lists
    ```c++
    class Solution {
    public:
        ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
            ListNode head(0);
            ListNode* p = &head;
            while(l1 && l2){
                if(l1->val < l2->val){
                    p->next = l1;
                    l1 = l1->next;
                }
                else{
                    p->next = l2;
                    l2 = l2->next;
                }
                p = p->next;
            }
            p->next = l1 ? l1 : l2;
            return head.next;
        }
    };
    ```

+ 160 Intersection of Two Linked Lists

    要求时间复杂度O(n)，空间复杂度O(1)，先任选一个链表，将该链表首尾相接，此时就变成了判断链表中是否有环的问题。

+ 406 Queue Reconstruct by Height

    贪心。从队列中找到最大高度的人，那么将不会存在比它更高的人，按k从低到高排，就是他们在队列的相对顺序。然后，将排好的人剔除，从剩余人的队列中，再找出最高的，直至队列为空。
    输入：[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]
    第一步：[[7,0],[7,1]]
    next：[[7,0],[6,1],[7,1]]
    next：[[5,0][7,0][5,2],[6,1],[7,1]]
    end：[[5,0][7,0][5,2],[6,1],[4,4],[7,1]]
    ```c++
    class Solution {
    public:
        
        vector<pair<int, int>> reconstructQueue(vector<pair<int, int>>& people) {
            sort(people.begin(), people.end(), [](const pair<int, int>& a, const pair<int, int>& b){
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            vector<pair<int, int>> res;
            for(auto person : people){
                res.insert(res.begin() + person.second, person);
            }
            return res;
        }
    };
    ```