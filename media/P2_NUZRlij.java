import java.io.*;


class P2
{
	public static void main(String args[])
	{
		Console c = System.console();

		int n = Integer.parseInt(c.readLine("Enter the number of line:"));

		if(n>0)
		{
			for(int i = 1; i <= n; i++)
			{
				for(int j = 1; j <= i; j++)
					System.out.print("* ");

				System.out.println();
			}

		}
		else
			System.out.println("Invalid input");

	}

}