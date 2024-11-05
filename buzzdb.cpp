#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <map>
#include <vector>
#include <list>
#include <unordered_map>
#include <string>
#include <memory>
#include <sstream>
#include <limits>
#include <thread>
#include <queue>
#include <optional>
#include <random>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <cstring>
#include <exception>
#include <atomic>
#include <set>

#define UNUSED(p) ((void)(p))

#define ASSERT_WITH_MESSAGE(condition, message)                                                                                  \
    do                                                                                                                           \
    {                                                                                                                            \
        if (!(condition))                                                                                                        \
        {                                                                                                                        \
            std::cerr << "Assertion \033[1;31mFAILED\033[0m: " << message << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::abort();                                                                                                        \
        }                                                                                                                        \
    } while (0)

enum FieldType
{
    INT,
    FLOAT,
    STRING,
    VECTOR
};

// Define a basic Field variant class that can hold different types
class Field
{
public:
    FieldType type;
    std::unique_ptr<char[]> data;
    size_t data_length;

public:
    Field(int i) : type(INT)
    {
        data_length = sizeof(int);
        data = std::make_unique<char[]>(data_length);
        std::memcpy(data.get(), &i, data_length);
    }

    Field(float f) : type(FLOAT)
    {
        data_length = sizeof(float);
        data = std::make_unique<char[]>(data_length);
        std::memcpy(data.get(), &f, data_length);
    }

    Field(const std::string &s) : type(STRING)
    {
        data_length = s.size() + 1; // include null-terminator
        data = std::make_unique<char[]>(data_length);
        std::memcpy(data.get(), s.c_str(), data_length);
    }

    Field(const std::vector<float> &vec) : type(VECTOR)
    {
        data_length = sizeof(uint32_t) + vec.size() * sizeof(float);
        data = std::make_unique<char[]>(data_length);

        uint32_t dim = vec.size();
        std::memcpy(data.get(), &dim, sizeof(uint32_t));

        std::memcpy(data.get() + sizeof(uint32_t), vec.data(),
                    vec.size() * sizeof(float));
    }

    std::vector<float> asVector() const {
        ASSERT_WITH_MESSAGE(type == VECTOR, "Field is not a vector");
        uint32_t dim;
        std::memcpy(&dim, data.get(), sizeof(uint32_t));
        
        std::vector<float> result(dim);
        std::memcpy(result.data(), data.get() + sizeof(uint32_t), 
                   dim * sizeof(float));
        return result;
    }

    Field &operator=(const Field &other)
    {
        if (&other == this)
        {
            return *this;
        }
        type = other.type;
        data_length = other.data_length;
        std::memcpy(data.get(), other.data.get(), data_length);
        return *this;
    }

    Field(Field &&other)
    {
        type = other.type;
        data_length = other.data_length;
        std::memcpy(data.get(), other.data.get(), data_length);
    }

    FieldType getType() const { return type; }
    int asInt() const
    {
        return *reinterpret_cast<int *>(data.get());
    }
    float asFloat() const
    {
        return *reinterpret_cast<float *>(data.get());
    }
    std::string asString() const
    {
        return std::string(data.get());
    }

    std::string serialize()
    {
        std::stringstream buffer;
        buffer << type << ' ' << data_length << ' ';
        if (type == STRING)
        {
            buffer << data.get() << ' ';
        }
        else if (type == INT)
        {
            buffer << *reinterpret_cast<int *>(data.get()) << ' ';
        }
        else if (type == FLOAT)
        {
            buffer << *reinterpret_cast<float *>(data.get()) << ' ';
        } else if (type == VECTOR) {
            auto vec = asVector();
            buffer << vec.size() << ' ';
            for (const auto& val : vec) {
                buffer << val << ' ';
            } ' ';
        }
        return buffer.str();
    }

    void serialize(std::ofstream &out)
    {
        std::string serializedData = this->serialize();
        out << serializedData;
    }

    static std::unique_ptr<Field> deserialize(std::istream &in)
    {
        int type;
        in >> type;
        size_t length;
        in >> length;
        if (type == STRING)
        {
            std::string val;
            in >> val;
            return std::make_unique<Field>(val);
        }
        else if (type == INT)
        {
            int val;
            in >> val;
            return std::make_unique<Field>(val);
        }
        else if (type == FLOAT)
        {
            float val;
            in >> val;
            return std::make_unique<Field>(val);
        }
        else if (type == VECTOR) {
            size_t dim;
            in >> dim;
            std::vector<float> vec(dim);
            for (size_t i = 0; i < dim; i++) {
                in >> vec[i];
            }
            return std::make_unique<Field>(vec);
        }
        return nullptr;
    }

    void print() const
    {
        switch (getType())
        {
        case INT:
            std::cout << asInt();
            break;
        case FLOAT:
            std::cout << asFloat();
            break;
        case STRING:
            std::cout << asString();
            break;
        case VECTOR:
            auto vec = asVector();
            std::cout << "[";
            for (size_t i = 0; i < vec.size(); i++) {
                std::cout << vec[i] << " ";
            }
            std::cout << "]";
            break;
        }
    }
};

class VectorIndex {
    private:
    struct Node {
        std::vector<float> center;
        std::vector<size_t> point_ids;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        size_t split_dim;
        
        Node(const std::vector<float>& center) 
            : center(center), split_dim(0) {}
    };

    std::unique_ptr<Node> root;
    size_t dimensions;
    static constexpr size_t MAX_POINTS_PER_NODE = 10;
    BufferManager& buffer_manager;

    public:
    VectorIndex(size_t dims, BufferManager& bm) 
        : dimensions(dims), buffer_manager(bm) {}

    void insert(const std::vector<float>& point, size_t point_id) {
        if (!root) {
            root = std::make_unique<Node>(point);
            root->point_ids.push_back(point_id);
            return;
        }
        insertRecursive(root.get(), point, point_id);
    }

    std::vector<size_t> knnSearch(const std::vector<float>& query, size_t k) {
        std::priority_queue<std::pair<float, size_t>> results;
        knnSearchRecursive(root.get(), query, k, results);
        
        std::vector<size_t> neighbors;
        while (!results.empty()) {
            neighbors.push_back(results.top().second);
            results.pop();
        }
        std::reverse(neighbors.begin(), neighbors.end());
        return neighbors;
    }
    private:
    void insertRecursive(Node* node, const std::vector<float>& point, 
                        size_t point_id) {
        if (node->point_ids.size() < MAX_POINTS_PER_NODE) {
            node->point_ids.push_back(point_id);
            return;
        }

        if (!node->left) {
            float split_value = node->center[node->split_dim];
            
            if (point[node->split_dim] < split_value) {
                node->left = std::make_unique<Node>(point);
                node->left->split_dim = (node->split_dim + 1) % dimensions;
                node->left->point_ids.push_back(point_id);
            } else {
                node->right = std::make_unique<Node>(point);
                node->right->split_dim = (node->split_dim + 1) % dimensions;
                node->right->point_ids.push_back(point_id);
            }
            return;
        }

        if (point[node->split_dim] < node->center[node->split_dim]) {
            insertRecursive(node->left.get(), point, point_id);
        } else {
            insertRecursive(node->right.get(), point, point_id);
        }
    }

    void knnSearchRecursive(Node* node, const std::vector<float>& query, 
                           size_t k, 
                           std::priority_queue<std::pair<float, size_t>>& results) {
        if (!node) return;

        for (size_t id : node->point_ids) {
            float dist = computeDistance(query, node->center);
            if (results.size() < k) {
                results.push({dist, id});
            } else if (dist < results.top().first) {
                results.pop();
                results.push({dist, id});
            }
        }

        float diff = query[node->split_dim] - node->center[node->split_dim];
        if (diff < 0) {
            knnSearchRecursive(node->left.get(), query, k, results);
            if (results.size() < k || std::abs(diff) < results.top().first) {
                knnSearchRecursive(node->right.get(), query, k, results);
            }
        } else {
            knnSearchRecursive(node->right.get(), query, k, results);
            if (results.size() < k || std::abs(diff) < results.top().first) {
                knnSearchRecursive(node->left.get(), query, k, results);
            }
        }
    }

    float computeDistance(const std::vector<float>& a, 
                         const std::vector<float>& b) {
        float dist = 0.0f;
        for (size_t i = 0; i < dimensions; i++) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
};

class Tuple
{
public:
    std::vector<std::unique_ptr<Field>> fields;

    void addField(std::unique_ptr<Field> field)
    {
        fields.push_back(std::move(field));
    }

    size_t getSize() const
    {
        size_t size = 0;
        for (const auto &field : fields)
        {
            size += field->data_length;
        }
        return size;
    }

    std::string serialize()
    {
        std::stringstream buffer;
        buffer << fields.size() << ' ';
        for (const auto &field : fields)
        {
            buffer << field->serialize();
        }
        return buffer.str();
    }

    void serialize(std::ofstream &out)
    {
        std::string serializedData = this->serialize();
        out << serializedData;
    }

    static std::unique_ptr<Tuple> deserialize(std::istream &in)
    {
        auto tuple = std::make_unique<Tuple>();
        size_t fieldCount;
        in >> fieldCount;
        for (size_t i = 0; i < fieldCount; ++i)
        {
            tuple->addField(Field::deserialize(in));
        }
        return tuple;
    }

    void print() const
    {
        for (const auto &field : fields)
        {
            field->print();
            std::cout << " ";
        }
        std::cout << "\n";
    }
};

static constexpr size_t PAGE_SIZE = 4096;                      // Fixed page size
static constexpr size_t MAX_SLOTS = 512;                       // Fixed number of slots
static constexpr size_t MAX_PAGES = 1000;                      // Total Number of pages that can be stored
uint16_t INVALID_VALUE = std::numeric_limits<uint16_t>::max(); // Sentinel value

struct Slot
{
    bool empty = true;               // Is the slot empty?
    uint16_t offset = INVALID_VALUE; // Offset of the slot within the page
    uint16_t length = INVALID_VALUE; // Length of the slot
};

// Slotted Page class
class SlottedPage
{
public:
    std::unique_ptr<char[]> page_data = std::make_unique<char[]>(PAGE_SIZE);
    size_t metadata_size = sizeof(Slot) * MAX_SLOTS;

    SlottedPage()
    {
        // Empty page -> initialize slot array inside page
        Slot *slot_array = reinterpret_cast<Slot *>(page_data.get());
        for (size_t slot_itr = 0; slot_itr < MAX_SLOTS; slot_itr++)
        {
            slot_array[slot_itr].empty = true;
            slot_array[slot_itr].offset = INVALID_VALUE;
            slot_array[slot_itr].length = INVALID_VALUE;
        }
    }

    // Add a tuple, returns true if it fits, false otherwise.
    bool addTuple(std::unique_ptr<Tuple> tuple)
    {

        // Serialize the tuple into a char array
        auto serializedTuple = tuple->serialize();
        size_t tuple_size = serializedTuple.size();

        // std::cout << "Tuple size: " << tuple_size << " bytes\n";
        assert(tuple_size == 38);

        // Check for first slot with enough space
        size_t slot_itr = 0;
        Slot *slot_array = reinterpret_cast<Slot *>(page_data.get());
        for (; slot_itr < MAX_SLOTS; slot_itr++)
        {
            if (slot_array[slot_itr].empty == true and
                slot_array[slot_itr].length >= tuple_size)
            {
                break;
            }
        }
        if (slot_itr == MAX_SLOTS)
        {
            // std::cout << "Page does not contain an empty slot with sufficient space to store the tuple.";
            return false;
        }

        // Identify the offset where the tuple will be placed in the page
        // Update slot meta-data if needed
        slot_array[slot_itr].empty = false;
        size_t offset = INVALID_VALUE;
        if (slot_array[slot_itr].offset == INVALID_VALUE)
        {
            if (slot_itr != 0)
            {
                auto prev_slot_offset = slot_array[slot_itr - 1].offset;
                auto prev_slot_length = slot_array[slot_itr - 1].length;
                offset = prev_slot_offset + prev_slot_length;
            }
            else
            {
                offset = metadata_size;
            }

            slot_array[slot_itr].offset = offset;
        }
        else
        {
            offset = slot_array[slot_itr].offset;
        }

        if (offset + tuple_size >= PAGE_SIZE)
        {
            slot_array[slot_itr].empty = true;
            slot_array[slot_itr].offset = INVALID_VALUE;
            return false;
        }

        assert(offset != INVALID_VALUE);
        assert(offset >= metadata_size);
        assert(offset + tuple_size < PAGE_SIZE);

        if (slot_array[slot_itr].length == INVALID_VALUE)
        {
            slot_array[slot_itr].length = tuple_size;
        }

        // Copy serialized data into the page
        std::memcpy(page_data.get() + offset,
                    serializedTuple.c_str(),
                    tuple_size);

        return true;
    }

    void deleteTuple(size_t index)
    {
        Slot *slot_array = reinterpret_cast<Slot *>(page_data.get());
        size_t slot_itr = 0;
        for (; slot_itr < MAX_SLOTS; slot_itr++)
        {
            if (slot_itr == index and
                slot_array[slot_itr].empty == false)
            {
                slot_array[slot_itr].empty = true;
                break;
            }
        }

        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void print() const
    {
        Slot *slot_array = reinterpret_cast<Slot *>(page_data.get());
        for (size_t slot_itr = 0; slot_itr < MAX_SLOTS; slot_itr++)
        {
            if (slot_array[slot_itr].empty == false)
            {
                assert(slot_array[slot_itr].offset != INVALID_VALUE);
                const char *tuple_data = page_data.get() + slot_array[slot_itr].offset;
                std::istringstream iss(tuple_data);
                auto loadedTuple = Tuple::deserialize(iss);
                std::cout << "Slot " << slot_itr << " : [";
                std::cout << (uint16_t)(slot_array[slot_itr].offset) << "] :: ";
                loadedTuple->print();
            }
        }
        std::cout << "\n";
    }
};

const std::string database_filename = "buzzdb.dat";

class StorageManager
{
public:
    std::fstream fileStream;
    size_t num_pages = 0;
    std::mutex io_mutex;

public:
    StorageManager(bool truncate_mode = true)
    {
        auto flags = truncate_mode ? std::ios::in | std::ios::out | std::ios::trunc
                                   : std::ios::in | std::ios::out;
        fileStream.open(database_filename, flags);
        if (!fileStream)
        {
            // If file does not exist, create it
            fileStream.clear(); // Reset the state
            fileStream.open(database_filename, truncate_mode ? (std::ios::out | std::ios::trunc) : std::ios::out);
        }
        fileStream.close();
        fileStream.open(database_filename, std::ios::in | std::ios::out);

        fileStream.seekg(0, std::ios::end);
        num_pages = fileStream.tellg() / PAGE_SIZE;

        if (num_pages == 0)
        {
            extend();
        }
    }

    ~StorageManager()
    {
        if (fileStream.is_open())
        {
            fileStream.close();
        }
    }

    // Read a page from disk
    std::unique_ptr<SlottedPage> load(uint16_t page_id)
    {
        fileStream.seekg(page_id * PAGE_SIZE, std::ios::beg);
        auto page = std::make_unique<SlottedPage>();
        // Read the content of the file into the page
        if (fileStream.read(page->page_data.get(), PAGE_SIZE))
        {
            // std::cout << "Page read successfully from file." << std::endl;
        }
        else
        {
            std::cerr << "Error: Unable to read data from the file. \n";
            exit(-1);
        }
        return page;
    }

    // Write a page to disk
    void flush(uint16_t page_id, const SlottedPage &page)
    {
        size_t page_offset = page_id * PAGE_SIZE;

        // Move the write pointer
        fileStream.seekp(page_offset, std::ios::beg);
        fileStream.write(page.page_data.get(), PAGE_SIZE);
        fileStream.flush();
    }

    // Extend database file by one page
    void extend()
    {
        // Create a slotted page
        auto empty_slotted_page = std::make_unique<SlottedPage>();

        // Move the write pointer
        fileStream.seekp(0, std::ios::end);

        // Write the page to the file, extending it
        fileStream.write(empty_slotted_page->page_data.get(), PAGE_SIZE);
        fileStream.flush();

        // Update number of pages
        num_pages += 1;
    }

    void extend(uint64_t till_page_id)
    {
        std::lock_guard<std::mutex> io_guard(io_mutex);
        uint64_t write_size = std::max(static_cast<uint64_t>(0), till_page_id + 1 - num_pages) * PAGE_SIZE;
        if (write_size > 0)
        {
            // std::cout << "Extending database file till page id : "<<till_page_id<<" \n";
            char *buffer = new char[write_size];
            std::memset(buffer, 0, write_size);

            fileStream.seekp(0, std::ios::end);
            fileStream.write(buffer, write_size);
            fileStream.flush();

            num_pages = till_page_id + 1;
        }
    }
};

using PageID = uint16_t;

class Policy
{
public:
    virtual bool touch(PageID page_id) = 0;
    virtual PageID evict() = 0;
    virtual ~Policy() = default;
};

void printList(std::string list_name, const std::list<PageID> &myList)
{
    std::cout << list_name << " :: ";
    for (const PageID &value : myList)
    {
        std::cout << value << ' ';
    }
    std::cout << '\n';
}

class LruPolicy : public Policy
{
private:
    // List to keep track of the order of use
    std::list<PageID> lruList;

    // Map to find a page's iterator in the list efficiently
    std::unordered_map<PageID, std::list<PageID>::iterator> map;

    size_t cacheSize;

public:
    LruPolicy(size_t cacheSize) : cacheSize(cacheSize) {}

    bool touch(PageID page_id) override
    {
        // printList("LRU", lruList);

        bool found = false;
        // If page already in the list, remove it
        if (map.find(page_id) != map.end())
        {
            found = true;
            lruList.erase(map[page_id]);
            map.erase(page_id);
        }

        // If cache is full, evict
        if (lruList.size() == cacheSize)
        {
            evict();
        }

        if (lruList.size() < cacheSize)
        {
            // Add the page to the front of the list
            lruList.emplace_front(page_id);
            map[page_id] = lruList.begin();
        }

        return found;
    }

    PageID evict() override
    {
        // Evict the least recently used page
        PageID evictedPageId = INVALID_VALUE;
        if (lruList.size() != 0)
        {
            evictedPageId = lruList.back();
            map.erase(evictedPageId);
            lruList.pop_back();
        }
        return evictedPageId;
    }
};

constexpr size_t MAX_PAGES_IN_MEMORY = 10;

class BufferManager
{
private:
    using PageMap = std::unordered_map<PageID, SlottedPage>;

    StorageManager storage_manager;
    PageMap pageMap;
    std::unique_ptr<Policy> policy;

public:
    BufferManager(bool storage_manager_truncate_mode = true) : storage_manager(storage_manager_truncate_mode),
                                                               policy(std::make_unique<LruPolicy>(MAX_PAGES_IN_MEMORY))
    {
        storage_manager.extend(MAX_PAGES);
    }

    ~BufferManager()
    {
        for (auto &pair : pageMap)
        {
            flushPage(pair.first);
        }
    }

    SlottedPage &fix_page(int page_id)
    {
        auto it = pageMap.find(page_id);
        if (it != pageMap.end())
        {
            policy->touch(page_id);
            return pageMap.find(page_id)->second;
        }

        if (pageMap.size() >= MAX_PAGES_IN_MEMORY)
        {
            auto evictedPageId = policy->evict();
            if (evictedPageId != INVALID_VALUE)
            {
                // std::cout << "Evicting page " << evictedPageId << "\n";
                storage_manager.flush(evictedPageId,
                                      pageMap[evictedPageId]);
            }
        }

        auto page = storage_manager.load(page_id);
        policy->touch(page_id);
        // std::cout << "Loading page: " << page_id << "\n";
        pageMap[page_id] = std::move(*page);
        return pageMap[page_id];
    }

    void flushPage(int page_id)
    {
        storage_manager.flush(page_id, pageMap[page_id]);
    }

    void extend()
    {
        storage_manager.extend();
    }

    size_t getNumPages()
    {
        return storage_manager.num_pages;
    }
};
uint64_t disk_page_counter = 1;
std::optional<uint64_t> disk_tree_root = std::nullopt;
template <typename KeyT, typename ValueT, typename ComparatorT, size_t PageSize>
class BTree
{
public:
    struct Node
    {
        /// The level in the tree.
        uint16_t level;

        /// The number of children.
        uint16_t count;

        // Constructor
        Node(uint16_t level, uint16_t count)
            : level(level), count(count) {}

        /// Is the node a leaf node?
        bool is_leaf() const { return level == 0; }
    };

    struct InnerNode : public Node
    {
        /// The capacity of a node.
        static constexpr uint32_t kCapacity = (PAGE_SIZE - sizeof(Node)) / (sizeof(KeyT) + sizeof(uint64_t));

        /// The keys.
        KeyT keys[kCapacity - 1];

        /// The children.
        uint64_t children[kCapacity];

        /// Constructor.
        InnerNode() : Node(0, 0) {}

        /// Get the index of the first key that is not less than than a provided key.
        /// @param[in] key          The key that should be searched.
        std::pair<uint32_t, bool> lower_bound(const KeyT &key)
        {
            ComparatorT comp;
            uint32_t idx = 0;
            uint32_t node_count = static_cast<uint32_t>(this->count);
            while (idx < node_count - 1 && comp(this->keys[idx], key))
                idx++;
            bool found = (idx < node_count - 1) && !comp(keys[idx], key) && !comp(key, keys[idx]);
            return {idx, found};
        }

        /// Insert a key.
        /// @param[in] key          The separator that should be inserted.
        /// @param[in] split_page   The id of the split page that should be inserted.
        void insert(const KeyT &key, uint64_t split_page)
        {
            auto [pos, found] = lower_bound(key);

            uint32_t node_count = static_cast<uint32_t>(this->count);

            if (pos < node_count - 1)
            {
                std::move_backward(keys + pos, keys + node_count - 1, keys + node_count);
                std::move_backward(children + pos + 1, children + node_count, children + node_count + 1);
            }

            keys[pos] = key;
            children[pos + 1] = split_page;
            this->count++;
        }

        /// Split the inner node.
        /// @param[in] inner_node       The inner node being split.
        /// @return                 The separator key.
        KeyT split(InnerNode *inner_node)
        {
            uint32_t mid = (this->count + 1) / 2;
            KeyT separator = keys[mid];

            inner_node->level = this->level;
            inner_node->count = this->count - mid - 1;

            std::copy(keys + mid + 1, keys + this->count, inner_node->keys);
            std::copy(children + mid + 1, children + this->count + 1, inner_node->children);

            this->count = mid + 1;

            return separator;
        }
    };

    struct LeafNode : public Node
    {
        /// The capacity of a node.
        static constexpr uint32_t kCapacity = (PAGE_SIZE - sizeof(Node)) / (sizeof(KeyT) + sizeof(ValueT));

        /// The keys.
        KeyT keys[kCapacity];

        /// The values.
        ValueT values[kCapacity];

        /// Constructor.
        LeafNode() : Node(0, 0) {}

        /// Insert a key.
        /// @param[in] key          The key that should be inserted.
        /// @param[in] value        The value that should be inserted.
        void insert(const KeyT &key, const ValueT &value)
        {
            ComparatorT comp;
            int32_t target_pos = this->count;

            int32_t left = 0, right = this->count;
            while (left < right)
            {
                int32_t mid = left + (right - left) / 2;
                if (comp(keys[mid], key))
                    left = mid + 1;
                else
                    right = mid;
            }
            target_pos = left;

            if (target_pos < this->count && !comp(keys[target_pos], key) && !comp(key, keys[target_pos]))
            {
                values[target_pos] = value;
                return;
            }

            if (target_pos < this->count)
            {
                std::move_backward(keys + target_pos, keys + this->count, keys + this->count + 1);
                std::move_backward(values + target_pos, values + this->count, values + this->count + 1);
            }

            keys[target_pos] = key;
            values[target_pos] = value;
            this->count++;
        }

        /// Erase a key.
        void erase(const KeyT &key)
        {
            ComparatorT comparator;
            uint32_t pos = 0;

            while (pos < this->count && comparator(keys[pos], key))
            {
                pos++;
            }

            if (pos < this->count && !comparator(keys[pos], key))
            {
                std::move(keys + pos + 1, keys + this->count, keys + pos);
                std::move(values + pos + 1, values + this->count, values + pos);
                this->count--;
            }
        }

        /// Split the leaf node.
        /// @param[in] leaf_node       The leaf node being split
        /// @return                 The separator key.
        KeyT split(LeafNode *leaf_node)
        {
            // std::cout << "Starting split operation..." << std::endl;
            uint32_t split_idx = this->count / 2;
            leaf_node->count = this->count - split_idx;
            leaf_node->level = this->level;

            std::copy(this->keys + split_idx, this->keys + this->count, leaf_node->keys);
            std::copy(this->values + split_idx, this->values + this->count, leaf_node->values);

            this->count = split_idx;
            return leaf_node->keys[0];
        }
    };

    /// The root.
    std::optional<uint64_t> root;

    /// The buffer manager
    BufferManager &buffer_manager;

    /// Next page id.
    /// You don't need to worry about about the page allocation.
    /// (Neither fragmentation, nor persisting free-space bitmaps)
    /// Just increment the next_page_id whenever you need a new page.
    uint64_t next_page_id;

    /// Constructor.
    BTree(BufferManager &buffer_manager) : buffer_manager(buffer_manager)
    {
        next_page_id = disk_page_counter;
    }

    /// Lookup an entry in the tree.
    /// @param[in] key      The key that should be searched.
    std::optional<ValueT> lookup(const KeyT &key)
    {
        if (!(root = disk_tree_root))
        {
            return std::nullopt;
        }

        uint64_t current_page_id = *root;
        while (true)
        {
            auto &page = buffer_manager.fix_page(current_page_id);
            Node *node = reinterpret_cast<Node *>(page.page_data.get());

            if (node->is_leaf())
            {
                LeafNode *leaf = static_cast<LeafNode *>(node);
                int left = 0, right = leaf->count - 1;

                while (left <= right)
                {
                    int mid = left + (right - left) / 2;
                    if (ComparatorT{}(leaf->keys[mid], key))
                        left = mid + 1;
                    else if (ComparatorT{}(key, leaf->keys[mid]))
                        right = mid - 1;
                    else
                        return leaf->values[mid];
                }
                return std::nullopt;
            }
            else
            {
                InnerNode *inner = static_cast<InnerNode *>(node);
                int idx = inner->count - 1;
                while (idx > 0 && ComparatorT{}(key, inner->keys[idx - 1]))
                    idx--;
                current_page_id = inner->children[idx];
            }
        }
    }

    /// Erase an entry in the tree.
    /// @param[in] key      The key that should be searched.
    void erase(const KeyT &key)
    {
        if (!root)
            return;

        uint64_t current_page_id = *root;
        std::vector<uint64_t> path;
        while (true)
        {
            path.push_back(current_page_id);
            auto &page = buffer_manager.fix_page(current_page_id);
            Node *node = reinterpret_cast<Node *>(page.page_data.get());

            if (node->is_leaf())
            {
                LeafNode *leaf = static_cast<LeafNode *>(node);
                leaf->erase(key);
                return;
            }
            auto inner = static_cast<InnerNode *>(node);
            ComparatorT comp;
            uint32_t idx = 0;
            idx = std::find_if(inner->keys, inner->keys + inner->count - 1, [&](const KeyT &k)
                               { return comp(key, k); }) -
                  inner->keys;
            current_page_id = inner->children[idx];
        }
    }

    /// Inserts a new entry into the tree.
    /// @param[in] key      The key that should be inserted.
    /// @param[in] value    The value that should be inserted.
    void insert(const KeyT &key, const ValueT &value)
    {
        // std::cout << "Inserting key: " << key << ", value: " << value << std::endl;
        if (!root)
        {
            disk_tree_root = root = next_page_id++;
            disk_page_counter = next_page_id;

            auto &page = buffer_manager.fix_page(*root);
            auto leaf = new (page.page_data.get()) LeafNode();
            leaf->insert(key, value);
            // std::cout << "Created new root (leaf) with key " << key << std::endl;
            return;
        }

        uint64_t current_page_id = *root;
        std::vector<uint64_t> path;

        while (true)
        {
            path.push_back(current_page_id);
            auto &page = buffer_manager.fix_page(current_page_id);
            Node *node = reinterpret_cast<Node *>(page.page_data.get());
            // std::cout << "Visiting node at level " << node->level << " with count " << node->count << std::endl;

            if (node->is_leaf())
            {
                LeafNode *leaf = static_cast<LeafNode *>(node);
                // std::cout << "Found leaf node with count " << leaf->count << std::endl;

                if (leaf->count < LeafNode::kCapacity)
                {
                    leaf->insert(key, value);
                    // std::cout << "Inserted into leaf node, new count: " << leaf->count << std::endl;
                    return;
                }

                // std::cout << "Leaf node full, splitting..." << std::endl;
                uint64_t new_page_id = next_page_id++;
                disk_page_counter = next_page_id;
                auto &new_page = buffer_manager.fix_page(new_page_id);
                auto new_leaf = new (new_page.page_data.get()) LeafNode();

                new_leaf->level = 0;

                KeyT separator = leaf->split(new_leaf);
                // std::cout << "Split leaf node, separator key: " << separator << std::endl;

                if (ComparatorT{}(key, separator))
                {
                    leaf->insert(key, value);
                    // std::cout << "Inserted key into original leaf" << std::endl;
                }
                else
                {
                    new_leaf->insert(key, value);
                    // std::cout << "Inserted key into new leaf" << std::endl;
                }

                if (path.size() == 1)
                {
                    uint64_t new_root_id = next_page_id++;
                    auto &new_root_page = buffer_manager.fix_page(new_root_id);
                    auto new_root = new (new_root_page.page_data.get()) InnerNode();
                    new_root->level = leaf->level + 1;
                    new_root->count = 2;
                    new_root->children[0] = current_page_id;
                    new_root->keys[0] = separator;
                    new_root->children[1] = new_page_id;
                    disk_tree_root = root = new_root_id;
                    disk_page_counter = next_page_id;
                    // std::cout << "Created new root (inner) with separator " << separator << std::endl;
                    return;
                }

                uint64_t new_child_id = new_page_id;
                KeyT current_separator = separator;
                while (!path.empty())
                {
                    path.pop_back();
                    if (path.empty())
                    {
                        uint64_t new_root_id = next_page_id++;
                        auto &new_root_page = buffer_manager.fix_page(new_root_id);
                        auto new_root = new (new_root_page.page_data.get()) InnerNode();

                        new_root->level = 1;
                        new_root->count = 2;
                        new_root->children[0] = current_page_id;
                        new_root->keys[0] = current_separator;
                        new_root->children[1] = new_child_id;

                        disk_tree_root = root = new_root_id;
                        disk_page_counter = next_page_id;
                        return;
                    }
                    current_page_id = path.back();
                    auto &parent_page = buffer_manager.fix_page(current_page_id);
                    auto parent = static_cast<InnerNode *>(reinterpret_cast<Node *>(parent_page.page_data.get()));
                    // std::cout << "Propagating split, current parent level: " << parent->level << ", count: " << parent->count << std::endl;

                    if (parent->count < InnerNode::kCapacity)
                    {
                        parent->insert(current_separator, new_child_id);
                        // std::cout << "Inserted into parent, new count: " << parent->count << std::endl;
                        return;
                    }

                    // std::cout << "Parent node full, splitting inner node..." << std::endl;
                    uint64_t new_inner_page_id = next_page_id++;
                    disk_page_counter = next_page_id;
                    auto &new_inner_page = buffer_manager.fix_page(new_inner_page_id);
                    auto new_inner = new (new_inner_page.page_data.get()) InnerNode();
                    new_inner->level = parent->level;

                    KeyT new_separator = parent->split(new_inner);
                    (ComparatorT{}(current_separator, new_separator) ? parent : new_inner)->insert(current_separator, new_child_id);

                    current_separator = new_separator;
                    new_child_id = new_inner_page_id;
                }

                return;
            }
            else
            {
                InnerNode *inner = static_cast<InnerNode *>(node);
                ComparatorT comp;
                uint32_t idx = 0;
                idx = std::find_if(inner->keys, inner->keys + inner->count - 1, [&](const KeyT &k)
                                   { return comp(key, k); }) -
                      inner->keys;
                current_page_id = inner->children[idx];
            }
        }
    }
};

int main(int argc, char *argv[])
{
    bool execute_all = false;
    std::string selected_test = "-1";

    if (argc < 2)
    {
        execute_all = true;
    }
    else
    {
        selected_test = argv[1];
    }

    using BTree = BTree<uint64_t, uint64_t, std::less<uint64_t>, 1024>;

    // Test 1: InsertEmptyTree
    if (execute_all || selected_test == "1")
    {
        std::cout << "...Starting Test 1" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);

        ASSERT_WITH_MESSAGE(tree.root.has_value() == false,
                            "tree.root is not nullptr");

        tree.insert(42, 21);

        ASSERT_WITH_MESSAGE(tree.root.has_value(),
                            "tree.root is still nullptr after insertion");

        std::string test = "inserting an element into an empty B-Tree";

        // Fix root page and obtain root node pointer
        SlottedPage *root_page = &buffer_manager.fix_page(*tree.root);
        auto root_node = reinterpret_cast<BTree::Node *>(root_page->page_data.get());

        ASSERT_WITH_MESSAGE(root_node->is_leaf() == true,
                            test + " does not create a leaf node.");
        ASSERT_WITH_MESSAGE(root_node->count == 1,
                            test + " does not create a leaf node with count = 1.");

        std::cout << "\033[1m\033[32mPassed: Test 1\033[0m" << std::endl;
    }

    // Test 2: InsertLeafNode
    if (execute_all || selected_test == "2")
    {
        std::cout << "...Starting Test 2" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);

        ASSERT_WITH_MESSAGE(tree.root.has_value() == false,
                            "tree.root is not nullptr");

        for (auto i = 0ul; i < BTree::LeafNode::kCapacity; ++i)
        {
            tree.insert(i, 2 * i);
        }
        ASSERT_WITH_MESSAGE(tree.root.has_value(),
                            "tree.root is still nullptr after insertion");

        std::string test = "inserting BTree::LeafNode::kCapacity elements into an empty B-Tree";

        SlottedPage *root_page = &buffer_manager.fix_page(*tree.root);
        auto root_node = reinterpret_cast<BTree::Node *>(root_page->page_data.get());
        auto root_inner_node = static_cast<BTree::InnerNode *>(root_node);

        ASSERT_WITH_MESSAGE(root_node->is_leaf() == true,
                            test + " creates an inner node as root.");
        ASSERT_WITH_MESSAGE(root_inner_node->count == BTree::LeafNode::kCapacity,
                            test + " does not store all elements.");

        std::cout << "\033[1m\033[32mPassed: Test 2\033[0m" << std::endl;
    }

    // Test 3: InsertLeafNodeSplit
    if (execute_all || selected_test == "3")
    {
        std::cout << "...Starting Test 3" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);

        ASSERT_WITH_MESSAGE(tree.root.has_value() == false,
                            "tree.root is not nullptr");

        for (auto i = 0ul; i < BTree::LeafNode::kCapacity; ++i)
        {
            tree.insert(i, 2 * i);
        }
        ASSERT_WITH_MESSAGE(tree.root.has_value(),
                            "tree.root is still nullptr after insertion");

        SlottedPage *root_page = &buffer_manager.fix_page(*tree.root);
        auto root_node = reinterpret_cast<BTree::Node *>(root_page->page_data.get());
        auto root_inner_node = static_cast<BTree::InnerNode *>(root_node);

        assert(root_inner_node->is_leaf());
        assert(root_inner_node->count == BTree::LeafNode::kCapacity);

        // Let there be a split...
        tree.insert(424242, 42);

        std::string test =
            "inserting BTree::LeafNode::kCapacity + 1 elements into an empty B-Tree";

        ASSERT_WITH_MESSAGE(tree.root.has_value() != false, test + " removes the root :-O");

        SlottedPage *root_page1 = &buffer_manager.fix_page(*tree.root);
        root_node = reinterpret_cast<BTree::Node *>(root_page1->page_data.get());
        root_inner_node = static_cast<BTree::InnerNode *>(root_node);

        ASSERT_WITH_MESSAGE(root_inner_node->is_leaf() == false,
                            test + " does not create a root inner node");
        ASSERT_WITH_MESSAGE(root_inner_node->count == 2,
                            test + " creates a new root with count != 2");

        std::cout << "\033[1m\033[32mPassed: Test 3\033[0m" << std::endl;
    }

    // Test 4: LookupEmptyTree
    if (execute_all || selected_test == "4")
    {
        std::cout << "...Starting Test 4" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);

        std::string test = "searching for a non-existing element in an empty B-Tree";

        ASSERT_WITH_MESSAGE(tree.lookup(42).has_value() == false,
                            test + " seems to return something :-O");

        std::cout << "\033[1m\033[32mPassed: Test 4\033[0m" << std::endl;
    }

    // Test 5: LookupSingleLeaf
    if (execute_all || selected_test == "5")
    {
        std::cout << "...Starting Test 5" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);

        // Fill one page
        for (auto i = 0ul; i < BTree::LeafNode::kCapacity; ++i)
        {
            tree.insert(i, 2 * i);
            ASSERT_WITH_MESSAGE(tree.lookup(i).has_value(),
                                "searching for the just inserted key k=" + std::to_string(i) + " yields nothing");
        }

        // Lookup all values
        for (auto i = 0ul; i < BTree::LeafNode::kCapacity; ++i)
        {
            auto v = tree.lookup(i);
            ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(i) + " is missing");
            ASSERT_WITH_MESSAGE(*v == 2 * i, "key=" + std::to_string(i) + " should have the value v=" + std::to_string(2 * i));
        }

        std::cout << "\033[1m\033[32mPassed: Test 5\033[0m" << std::endl;
    }

    // Test 6: LookupSingleSplit
    if (execute_all || selected_test == "6")
    {
        std::cout << "...Starting Test 6" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);

        // Insert values
        for (auto i = 0ul; i < BTree::LeafNode::kCapacity; ++i)
        {
            tree.insert(i, 2 * i);
        }

        tree.insert(BTree::LeafNode::kCapacity, 2 * BTree::LeafNode::kCapacity);
        ASSERT_WITH_MESSAGE(tree.lookup(BTree::LeafNode::kCapacity).has_value(),
                            "searching for the just inserted key k=" + std::to_string(BTree::LeafNode::kCapacity + 1) + " yields nothing");

        // Lookup all values
        for (auto i = 0ul; i < BTree::LeafNode::kCapacity + 1; ++i)
        {
            auto v = tree.lookup(i);
            ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(i) + " is missing");
            ASSERT_WITH_MESSAGE(*v == 2 * i,
                                "key=" + std::to_string(i) + " should have the value v=" + std::to_string(2 * i));
        }

        std::cout << "\033[1m\033[32mPassed: Test 6\033[0m" << std::endl;
    }

    // Test 7: LookupMultipleSplitsIncreasing
    if (execute_all || selected_test == "7")
    {
        std::cout << "...Starting Test 7" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);
        auto n = 40 * BTree::LeafNode::kCapacity;

        // Insert values
        for (auto i = 0ul; i < n; ++i)
        {
            tree.insert(i, 2 * i);
            ASSERT_WITH_MESSAGE(tree.lookup(i).has_value(),
                                "searching for the just inserted key k=" + std::to_string(i) + " yields nothing");
        }

        // Lookup all values
        for (auto i = 0ul; i < n; ++i)
        {
            auto v = tree.lookup(i);
            ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(i) + " is missing");
            ASSERT_WITH_MESSAGE(*v == 2 * i,
                                "key=" + std::to_string(i) + " should have the value v=" + std::to_string(2 * i));
        }
        std::cout << "\033[1m\033[32mPassed: Test 7\033[0m" << std::endl;
    }

    // Test 8: LookupMultipleSplitsDecreasing
    if (execute_all || selected_test == "8")
    {
        std::cout << "...Starting Test 8" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);
        auto n = 10 * BTree::LeafNode::kCapacity;

        // Insert values
        for (auto i = n; i > 0; --i)
        {
            tree.insert(i, 2 * i);
            ASSERT_WITH_MESSAGE(tree.lookup(i).has_value(),
                                "searching for the just inserted key k=" + std::to_string(i) + " yields nothing");
        }

        // Lookup all values
        for (auto i = n; i > 0; --i)
        {
            auto v = tree.lookup(i);
            ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(i) + " is missing");
            ASSERT_WITH_MESSAGE(*v == 2 * i,
                                "key=" + std::to_string(i) + " should have the value v=" + std::to_string(2 * i));
        }

        std::cout << "\033[1m\033[32mPassed: Test 8\033[0m" << std::endl;
    }

    // Test 9: LookupRandomNonRepeating
    if (execute_all || selected_test == "9")
    {
        std::cout << "...Starting Test 9" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);
        auto n = 10 * BTree::LeafNode::kCapacity;

        // Generate random non-repeating key sequence
        std::vector<uint64_t> keys(n);
        std::iota(keys.begin(), keys.end(), n);
        std::mt19937_64 engine(0);
        std::shuffle(keys.begin(), keys.end(), engine);

        // Insert values
        for (auto i = 0ul; i < n; ++i)
        {
            tree.insert(keys[i], 2 * keys[i]);
            ASSERT_WITH_MESSAGE(tree.lookup(keys[i]).has_value(),
                                "searching for the just inserted key k=" + std::to_string(keys[i]) +
                                    " after i=" + std::to_string(i) + " inserts yields nothing");
        }

        // Lookup all values
        for (auto i = 0ul; i < n; ++i)
        {
            auto v = tree.lookup(keys[i]);
            ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(keys[i]) + " is missing");
            ASSERT_WITH_MESSAGE(*v == 2 * keys[i],
                                "key=" + std::to_string(keys[i]) + " should have the value v=" + std::to_string(2 * keys[i]));
        }

        std::cout << "\033[1m\033[32mPassed: Test 9\033[0m" << std::endl;
    }

    // Test 10: LookupRandomRepeating
    if (execute_all || selected_test == "10")
    {
        std::cout << "...Starting Test 10" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);
        auto n = 10 * BTree::LeafNode::kCapacity;

        // Insert & updated 100 keys at random
        std::mt19937_64 engine{0};
        std::uniform_int_distribution<uint64_t> key_distr(0, 99);
        std::vector<uint64_t> values(100);

        for (auto i = 1ul; i < n; ++i)
        {
            uint64_t rand_key = key_distr(engine);
            values[rand_key] = i;
            tree.insert(rand_key, i);

            auto v = tree.lookup(rand_key);
            ASSERT_WITH_MESSAGE(v.has_value(),
                                "searching for the just inserted key k=" + std::to_string(rand_key) +
                                    " after i=" + std::to_string(i - 1) + " inserts yields nothing");
            ASSERT_WITH_MESSAGE(*v == i,
                                "overwriting k=" + std::to_string(rand_key) + " with value v=" + std::to_string(i) +
                                    " failed");
        }

        // Lookup all values
        for (auto i = 0ul; i < 100; ++i)
        {
            if (values[i] == 0)
            {
                continue;
            }
            auto v = tree.lookup(i);
            ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(i) + " is missing");
            ASSERT_WITH_MESSAGE(*v == values[i],
                                "key=" + std::to_string(i) + " should have the value v=" + std::to_string(values[i]));
        }

        std::cout << "\033[1m\033[32mPassed: Test 10\033[0m" << std::endl;
    }

    // Test 11: Erase
    if (execute_all || selected_test == "11")
    {
        std::cout << "...Starting Test 11" << std::endl;
        BufferManager buffer_manager;
        BTree tree(buffer_manager);

        // Insert values
        for (auto i = 0ul; i < 2 * BTree::LeafNode::kCapacity; ++i)
        {
            tree.insert(i, 2 * i);
        }

        // Iteratively erase all values
        for (auto i = 0ul; i < 2 * BTree::LeafNode::kCapacity; ++i)
        {
            ASSERT_WITH_MESSAGE(tree.lookup(i).has_value(), "k=" + std::to_string(i) + " was not in the tree");
            tree.erase(i);
            ASSERT_WITH_MESSAGE(!tree.lookup(i), "k=" + std::to_string(i) + " was not removed from the tree");
        }
        std::cout << "\033[1m\033[32mPassed: Test 11\033[0m" << std::endl;
    }

    // Test 12: Persistant Btree
    if (execute_all || selected_test == "12")
    {
        std::cout << "...Starting Test 12" << std::endl;
        unsigned long n = 10 * BTree::LeafNode::kCapacity;

        // Build a tree
        {
            BufferManager buffer_manager;
            BTree tree(buffer_manager);

            // Insert values
            for (auto i = 0ul; i < n; ++i)
            {
                tree.insert(i, 2 * i);
                ASSERT_WITH_MESSAGE(tree.lookup(i).has_value(),
                                    "searching for the just inserted key k=" + std::to_string(i) + " yields nothing");
            }

            // Lookup all values
            for (auto i = 0ul; i < n; ++i)
            {
                auto v = tree.lookup(i);
                ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(i) + " is missing");
                ASSERT_WITH_MESSAGE(*v == 2 * i,
                                    "key=" + std::to_string(i) + " should have the value v=" + std::to_string(2 * i));
            }
        }

        // recreate the buffer manager and check for existence of the tree
        {
            BufferManager buffer_manager(false);
            BTree tree(buffer_manager);

            // Lookup all values
            for (auto i = 0ul; i < n; ++i)
            {
                auto v = tree.lookup(i);
                ASSERT_WITH_MESSAGE(v.has_value(), "key=" + std::to_string(i) + " is missing");
                ASSERT_WITH_MESSAGE(*v == 2 * i,
                                    "key=" + std::to_string(i) + " should have the value v=" + std::to_string(2 * i));
            }
        }

        std::cout << "\033[1m\033[32mPassed: Test 12\033[0m" << std::endl;
    }

    return 0;
}
