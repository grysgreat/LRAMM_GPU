struct g_tensor{
    int rows;
    int cols;
    float mean;
    float var;
    char in;
    char out;
    void *data;
}