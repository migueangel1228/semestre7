#include <iostream>
#include <thread>
#include <mutex>

using namespace std;

int counter = 0;
mutex mutexCounter;

void increment() {
    for (int i = 0; i < 100000; ++i) {
        lock_guard<mutex> lock(mutexCounter);
        ++counter;
    }
}

int main() {
    thread thread1(increment);
    thread thread2(increment);

    thread1.join();
    thread2.join();

    cout << "Counter: " << counter << endl;
    return 0;
}