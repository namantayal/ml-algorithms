#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

#define Xrow 4092
#define Xcol 5
#define Xrow_test 1364

void display(vector<vector<int>> &matrix)
{
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            cout <<"\t"<<matrix[i][j]<<"\t";
        }
        cout << endl;
    }
}

int map_to_int(string data)
{
    if(data=="Slight-Right-Turn")
        return 0;
    else if(data=="Sharp-Right-Turn")
        return 1;
    else if(data=="Move-Forward")
        return 2;
    else
        return 3;
}

void read_csv(vector<vector<double>> &X, vector<int> &Y, vector<vector<double>> &X_test, vector<int> &Y_test)
{
    fstream fin;
    string data;
    fin.open("dataset.csv", ios::in);
    for (int i = 0; i < Xrow; i++)
    {
        int j;
        X[i][0] = 1;
        for (j = 1; j < Xcol; j++)
        {
            getline(fin, data, ',');
            X[i][j] = stof(data);
        }
        getline(fin, data, '\n');
        Y[i] = map_to_int(data);
    }

    for (int i = 0; i < Xrow_test; i++)
    {
        int j;
        X_test[i][0] = 1;
        for (j = 1; j < Xcol; j++)
        {
            getline(fin, data, ',');
            X_test[i][j] = stof(data);
        }
        getline(fin, data, '\n');
        Y_test[i] = map_to_int(data);
    }
    fin.close();
}

string map_to_answer(int value)
{
    if(value==0)
        return "Slight-Right-Turn";
    else if(value==1)
        return "Sharp-Right-Turn";
    else if(value==2)
        return "Move-Forward";
    else
        return "Slight Left";
}

void write_csv(vector<vector<double>> &X, vector<int> &Y_test, vector<int> &Y_hat)
{
    fstream fout;
    fout.open("result.csv", ios::out);
    fout<<"Sensor1,Sensor2,Sensor3,Sensor4,,Actual,Predicted\n";
    for(int i=0;i<X.size();i++)
    {
        for(int j=1;j<X[0].size();j++)
        {
            fout<<X[i][j]<<",";
        }
        int index = Y_test[i];
        fout<<","<<map_to_answer(index);
        index =Y_hat[i];
        fout<<","<<map_to_answer(index)<<"\n";
    }
    fout.close();
}

double distance(vector<double> &X1, vector<double> &X2)
{
    double sum=0;
    for(int i=0;i<Xcol;i++)
    {
        sum+=pow((X1[i]-X2[i]),2);
    }
    return sqrt(sum);
}

bool sortcol(const vector<double> &v1, const vector<double> &v2) 
{ 
    return v1[0] < v2[0]; 
} 

void confusion_matrix(vector<int> &Y_test, vector<int> &Y_hat)
{
    vector<vector<int>> matrix(Xcol-1, vector<int>(Xcol-1,0));
    for(int i=0;i<Y_test.size();i++)
    {
        int row = Y_hat[i];
        int col = Y_test[i];
        matrix[row][col]+=1;
    }
    cout<<"\n\n\tConfusion Matrix - \n\n";
    display(matrix);
   
}

void accuracy(vector<int> &Y_test, vector<int> &Y_hat)
{
    int count=0;
    for(int i=0;i<Y_test.size();i++)
    {
        if(Y_test[i]==Y_hat[i])
            count++;
    }
    cout<<"\nACCURACY - "<<((float)count/Xrow_test)*100<<"%";
}

void predict(vector<vector<double>> &X,vector<int> &Y,vector<vector<double>> &X_test,vector<int> &Y_hat,int k)
{
    int count=1; 
    for(int j=0;j<X_test.size();j++)
    {
        vector<vector<double>> dist(Xrow,vector<double>(2));
        for(int i=0;i<X.size();i++)
        {
            dist[i][0]=distance(X[i],X_test[j]);
            dist[i][1]=Y[i];
        }
        sort(dist.begin(),dist.end(),sortcol);

        vector<int>vote(Xcol,0);
        for(int i=0;i<k;i++)
        {
            vote[(int)dist[i][1]]++;
        }
        Y_hat[j] = max_element(vote.begin(), vote.end()) - vote.begin();
        
        if(j==(count*(X_test.size()/5)))
        {
            cout<<((float)j/Xrow_test)*100<<"% completed"<<endl;
            count++;
        }
    }
}

int main()
{
    vector<vector<double>> X(Xrow, vector<double>(Xcol));
    vector<int> Y(Xrow);
    vector<vector<double>> X_test(Xrow_test, vector<double>(Xcol));
    vector<int> Y_test(Xrow_test);
    vector<int> Y_hat(Xrow_test);

    read_csv(X, Y, X_test, Y_test);
    predict(X,Y,X_test,Y_hat,3);
    write_csv(X_test,Y_test,Y_hat);
    confusion_matrix(Y_test,Y_hat);
    accuracy(Y_test,Y_hat);
    
    return 0;
}