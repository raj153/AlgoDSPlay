using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2241. Design an ATM Machine
    https://leetcode.com/problems/design-an-atm-machine/description/

    */
    public class ATM
    {
        long[] denominations = { 20, 50, 100, 200, 500 };
        long[] stores;

        public ATM()
        {
            stores = new long[5];
        }

        public void Deposit(int[] banknotesCount)
        {
            for (int i = 0; i < 5; i++)
            {
                stores[i] += banknotesCount[i];
            }
        }

        public int[] Withdraw(int amount)
        {
            long[] result = new long[5];
            int index = 4;
            while (amount > 0 && index >= 0)
            {
                long numberOfNotes = Math.Min(amount / denominations[index], stores[index]);
                result[index] = numberOfNotes;
                amount -= (int)(numberOfNotes * denominations[index]);
                index--;
            }

            if (amount != 0)
            {
                return new int[] { -1 };
            }
            else
            {
                for (int i = 0; i < 5; i++)
                {
                    stores[i] -= result[i];
                }
                return Array.ConvertAll(result, item => (int)item);
            }
        }
    }
}