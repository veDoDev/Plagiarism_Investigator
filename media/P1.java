import java.io.*;


class P1
{
	public static void main(String args[])
	{
		Console c = System.console();

		int n = Integer.parseInt(c.readLine("Enter the number of line:"));

		if(n>0)
		{
			int i = 1;
			while(i <= n)
			{
				int j = 1;
				while(j<=i)
				{
					System.out.print("* ");
					j++;
				}
			System.out.println();
			i++;
			}

		}
		else
			System.out.println("Invalid input");

	}

}