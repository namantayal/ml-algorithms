#include <iostream>
#include <fstream>
#include <vector>
#include <bits/stdc++.h>

#define Xrow 3227
#define Xrow_test 1075
#define Xcol 4

using namespace std;

template <typename T1>
void display(vector<vector<T1>> &matrix)
{
    cout << endl
         << "RESULT: " << endl;
    for (int i = 0; i < matrix.size(); i++)
    {
        cout << endl;
        for (int j = 0; j < matrix[0].size(); j++)
        {
            cout << " " << matrix[i][j];
        }
    }
}

void read_csv(vector<vector<double>> &X, vector<int> &Y,vector<vector<double>> &X_test, vector<int> &Y_test)
{
    fstream fin;
    string data;
    fin.open("dataset.csv", ios::in);
    for (int i = 0; i < Xrow; i++)
    {
        int j;
        for (j = 0; j < Xcol; j++)
        {
            getline(fin, data, ',');
            X[i][j] = stof(data);
        }
        getline(fin, data, '\n');
        Y[i] = stoi(data);
    }
    for (int i = 0; i < Xrow_test; i++)
    {
        int j;
        for (j = 0; j < Xcol; j++)
        {
            getline(fin, data, ',');
            X_test[i][j] = stof(data);
        }
        getline(fin, data, '\n');
        Y_test[i] = stoi(data);
    }

    fin.close();
}

void read_estimate(vector<double> &beta, double &c)
{
    fstream fin;
    string data;
    fin.open("estimate.csv",ios::in);
    for(int i=0;i<beta.size();i++)
    {
        getline(fin,data,',');
        beta[i]=stof(data);
    }
    getline(fin,data,'\n');
    getline(fin,data,'\n');
    c=stof(data);

    fin.close();
}

string map_to_answer(int value)
{
    if(value>=0.5)
        return "Sharp-Right-Turn";
    else
        return "Move-Forward";
}

void write_csv(vector<vector<double>> &X, vector<int> &Y_test, vector<double> &Y_hat)
{
    fstream fout;
    fout.open("result.csv", ios::out);
    fout<<"Sensor1,Sensor2,Sensor3,Sensor4,,Actual,Predicted\n";
    for(int i=0;i<X.size();i++)
    {
        for(int j=0;j<X[0].size();j++)
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

void write_estimate(vector<double> &beta, double &c)
{
    fstream fout;
    fout.open("estimate.csv",ios::out);
    for(int i=0;i<beta.size();i++)
    {
        fout<<beta[i]<<',';
    }
    fout<<'\n';
    fout<<c<<'\n';
    fout.close();
}

void m_transpose(vector<vector<double>> &matrix, vector<vector<double>> &transpose)
{
    for (int i = 0; i < matrix.size(); i++)
    {
        for (int j = 0; j < matrix[0].size(); j++)
        {
            transpose[j][i] = matrix[i][j];
        }
    }
}

template<typename t1,typename t2>
void m_multiply(vector<vector<t1>> &m1, vector<t2> &m2, vector<double> &answer)
{
    //multiplying first and second matrix
    for (int i = 0; i < m1.size(); i++)
    {
        for (int j = 0; j < m1[0].size(); j++)
        {
            answer[i] += m1[i][j] * m2[j];
        }
    }
}

void sigmoid(vector<double> &matrix)
{
    for (int i = 0; i < matrix.size(); i++)
    {
        matrix[i] = 1 / (1 + exp(-matrix[i]));
    }
}

double cost(vector<int> &Y,vector<double> &Y_hat)
{
    double sum=0;
    for(int i=0;i<Y.size();i++)
    {
        if(Y_hat[i]==0)
            sum+= Y[i] + ((1-Y[i])*log(1-Y_hat[i]));
        else if(Y_hat[i]==1)
            sum+= (Y[i]*log(Y_hat[i])) + ((1-Y[i]));
        else
            sum+= (Y[i]*log(Y_hat[i])) + ((1-Y[i])*log(1-Y_hat[i]));
    }
    return -sum/Xrow;
}



void beta_derivative(vector<vector<double>> &X, vector<int> &Y, vector<double> &Y_hat, vector<double> &beta_d)
{
    vector<vector<double>> X_transpose(Xcol, vector<double>(Xrow));
    m_transpose(X,X_transpose);
    vector<double> Y_hat_Y(Xrow);

    for(int i=0;i<Y.size();i++)
    {
        Y_hat_Y[i]=Y_hat[i]-Y[i];
    }

    m_multiply(X_transpose,Y_hat_Y,beta_d);

    for(int i=0;i<beta_d.size();i++)
    {
        beta_d[i]/=Xrow;
    }

}

double c_derivative(vector<int> &Y,vector<double> &Y_hat)
{
    double sum=0;
    for(int i=0;i<Y.size();i++)
    {
        sum+=(Y_hat[i]-Y[i]);
    }
    return sum/Xrow;
}

void predict(vector<vector<double>> &X, vector<double> &beta, double c, vector<double> &Y_hat)
{
    m_multiply(X, beta, Y_hat);
    for (int i = 0; i < Y_hat.size(); i++)
    {
        Y_hat[i] += c;
    }
    
    sigmoid(Y_hat);
}

void estimate(vector<vector<double>> &X, vector<int> &Y, vector<double> &beta, double &c, double learning_rate, int iterations)
{
    vector<double> Y_hat(Xrow);
    double cost_v,c_d;
    vector<double> beta_d(Xcol);

    for (int i = 1; i <= iterations; i++)
    {
        predict(X, beta, c, Y_hat);
        cost_v = cost(Y, Y_hat);
        beta_derivative(X, Y, Y_hat, beta_d);
        c_d = c_derivative(Y, Y_hat);
        for (int i = 0; i < beta.size(); i++)
        {
            beta[i] -= learning_rate * beta_d[i];
        }
        c -= learning_rate*c_d;
        if(i%1000 == 0)
            cout<<"cost - "<<cost_v<<endl;
    }
}

void confusion_matrix(vector<int> &Y_test, vector<double> &Y_hat)
{
    vector<vector<int>> matrix(2, vector<int>(2,0));
    for(int i=0;i<Y_test.size();i++)
    {
        int row = Y_hat[i];
        int col = Y_test[i];
        matrix[row][col]+=1;
    }
    cout<<"\n\nConfusion Matrix - \n\n";
    cout<<"\t"<<matrix[0][0]<<"\t"<<matrix[0][1]<<"\n";
    cout<<"\t"<<matrix[1][0]<<"\t"<<matrix[1][1]<<"\n";
}

int main()
{
    vector<vector<double>> X(Xrow, vector<double>(Xcol));
    vector<int> Y(Xrow);
    vector<vector<double>> X_test(Xrow_test, vector<double>(Xcol));
    vector<int> Y_test(Xrow_test);
    vector<double> beta(Xcol, 1);
    double c=0;

    read_csv(X, Y,X_test,Y_test);
    
    
    vector<double> Y_hat(Xrow_test);
    int choice;
    cout<<"\n1. Training \n2. Prediction\n\n>>";
    cin>>choice;
    if(choice==1)
    {
        estimate(X, Y, beta, c, 100, 10000);
        predict(X_test,beta,c,Y_hat);
        write_estimate(beta,c);
        
    }
    else
    {
        read_estimate(beta,c);
        predict(X_test,beta,c,Y_hat);
    }
    
    int count=0;
    for(int i=0;i<Xrow_test;i++){
        if(Y_hat[i]>=0.5 && Y_test[i]==1){
            count++;
        }
        else if(Y_hat[i]<0.5 && Y_test[i]==0){
            count++;
        }
    }

    confusion_matrix(Y_test,Y_hat);
    cout<<endl<<"Accuracy - "<<((float)count/Xrow_test)*100<<"%"<<endl;
    
    write_csv(X_test,Y_test,Y_hat);
    return 0;
}
