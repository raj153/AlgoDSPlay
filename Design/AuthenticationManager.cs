using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1797. Design Authentication Manager
    https://leetcode.com/problems/design-authentication-manager/description/

    Time complexity of CountUnexpiredTokens: O(n)
    Other methods cost time: O(1)


    */
    public class AuthenticationManager
    {

        private Dictionary<string, int> tokenExpiry = new Dictionary<string, int>();
        private int tokenLife;

        public AuthenticationManager(int timeToLive)
        {
            tokenLife = timeToLive;
        }

        public void Generate(string tokenId, int currentTime)
        {
            tokenExpiry[tokenId] = tokenLife + currentTime;
        }

        public void Renew(string tokenId, int currentTime)
        {
            if (tokenExpiry.TryGetValue(tokenId, out int expiryTime) && expiryTime > currentTime)
            {
                tokenExpiry[tokenId] = tokenLife + currentTime;
            }
        }

        public int CountUnexpiredTokens(int currentTime)
        {
            tokenExpiry = new Dictionary<string, int>(tokenExpiry.Where(e => e.Value > currentTime));
            return tokenExpiry.Count;
        }
    }

}
