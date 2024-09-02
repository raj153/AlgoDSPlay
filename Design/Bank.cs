using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    2043. Simple Bank System
    https://leetcode.com/problems/simple-bank-system/description/

    */
    public class Bank
    {
        long[] balance;
        public Bank(long[] balance)
        {
            this.balance = balance;
        }

        public bool transfer(int account1, int account2, long money)
        {
            if (account2 > balance.Length || !withdraw(account1, money)) return false;
            return deposit(account2, money);
        }

        public bool deposit(int account, long money)
        {
            if (account > balance.Length) return false;
            balance[account - 1] += money;
            return true;
        }

        public bool withdraw(int account, long money)
        {
            if (account > balance.Length || balance[account - 1] < money) return false;
            balance[account - 1] -= money;
            return true;
        }


    }
}