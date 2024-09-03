using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AlgoDSPlay.Design
{
    /*
    1357. Apply Discount Every n Orders
    https://leetcode.com/problems/apply-discount-every-n-orders/description/

    */
    public class Cashier
    {
        private int count = 0;
        private int numberOfItems;
        private int discountPercentage;
        private Dictionary<int, int> priceList = new Dictionary<int, int>();

        public Cashier(int n, int discount, int[] products, int[] prices)
        {
            this.numberOfItems = n;
            this.discountPercentage = discount;
            for (int i = 0; i < products.Length; ++i)
            {
                priceList[products[i]] = prices[i];
            }
        }

        public double GetBill(int[] product, int[] amount)
        {
            double totalCost = 0.0d;
            for (int i = 0; i < product.Length; ++i)
            {
                totalCost += priceList[product[i]] * amount[i];
            }
            return totalCost * (++count % numberOfItems == 0 ? 1 - discountPercentage / 100d : 1);
        }
    }
    /**
 * Your Cashier object will be instantiated and called as such:
 * Cashier obj = new Cashier(n, discount, products, prices);
 * double param_1 = obj.GetBill(product,amount);
 */
}