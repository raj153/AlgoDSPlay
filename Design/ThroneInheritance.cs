using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1600. Throne Inheritance
    https://leetcode.com/problems/throne-inheritance/description/

    DFS Problem with convoluted wording
    
    */
    public class ThroneInheritance
    {
        HashSet<string> set;
        string _kingName;
        Dictionary<string, List<string>> map;
        public ThroneInheritance(string kingName)
        {
            set = new();
            map = new();
            _kingName = kingName;
        }

        public void Birth(string parentName, string childName)
        {
            if (!map.ContainsKey(parentName)) map[parentName] = new List<string>();
            map[parentName].Add(childName);
        }

        public void Death(string name)
        {
            set.Add(name);
        }

        public IList<string> GetInheritanceOrder()
        {
            List<string> res = new();
            void dfs(string cur)
            {
                if (!set.Contains(cur))
                    res.Add(cur);
                if (!map.ContainsKey(cur)) return;
                for (int i = 0; i < map[cur]?.Count; i++)
                    dfs(map[cur][i]);
            }
            dfs(_kingName);
            return res;
        }
    }

}