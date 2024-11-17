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
#include <cmath>
#include <random>
#include <mutex>
#include <shared_mutex>
#include <cassert>
#include <cstring>
#include <exception>
#include <atomic>
#include <set>

#ifdef __unix__
#include <unistd.h>
#endif

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

double getCurrentMemoryUsage() {
    #ifdef __APPLE__
        struct rusage rusage;
        getrusage(RUSAGE_SELF, &rusage);
        return static_cast<double>(rusage.ru_maxrss) * 1024;
    #else
        return 0.0;
    #endif
}

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
            buffer << std::fixed << std::setprecision(6);
            for (const auto& val : vec) {
                buffer << val << ' ';
            }
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

    static float computeL2Distance(const std::vector<float>& a, 
                                 const std::vector<float>& b) {
        ASSERT_WITH_MESSAGE(a.size() == b.size(), 
            "Vectors must have same dimension");
        float dist = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    // Add vector normalization
    static std::vector<float> normalizeVector(const std::vector<float>& vec) {
        float magnitude = 0.0f;
        for (float val : vec) {
            magnitude += val * val;
        }
        magnitude = std::sqrt(magnitude);
        
        if (magnitude == 0) return vec;
        
        std::vector<float> normalized(vec.size());
        for (size_t i = 0; i < vec.size(); i++) {
            normalized[i] = vec[i] / magnitude;
        }
        return normalized;
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

// Base class for vector indexes
class VectorIndex {
public:
    virtual ~VectorIndex() = default;
    
    virtual void insert(const std::vector<float>& point, size_t point_id) = 0;
    
    virtual std::vector<std::pair<size_t, float>> search(
        const std::vector<float>& query,
        size_t k,
        size_t ef = 50) const = 0;
        
    virtual std::vector<std::pair<size_t, float>> rangeSearch(
        const std::vector<float>& query,
        float radius) const {
        // Default implementation using k-NN search
        auto knn_results = search(query, 100);  // Get enough neighbors
        std::vector<std::pair<size_t, float>> range_results;
        for (const auto& result : knn_results) {
            if (result.second <= radius) {
                range_results.push_back(result);
            }
        }
        return range_results;
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

    std::unique_ptr<Tuple> clone() const {
        auto new_tuple = std::make_unique<Tuple>();
        for (const auto &field : fields) {
            switch (field->getType()) {
            case INT:
                new_tuple->addField(std::make_unique<Field>(field->asInt()));
                break;
            case FLOAT:
                new_tuple->addField(std::make_unique<Field>(field->asFloat()));
                break;
            case STRING:
                new_tuple->addField(std::make_unique<Field>(field->asString()));
                break;
            case VECTOR:
                new_tuple->addField(std::make_unique<Field>(field->asVector()));
                break;
            }
        }
        return new_tuple;
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
constexpr size_t VECTOR_DIMENSION = 128; 
constexpr size_t DEFAULT_M = 16;
constexpr size_t DEFAULT_EF_CONSTRUCTION = 200;
static const float LEVEL_MULTIPLIER = 1.0f / log(DEFAULT_M);
constexpr size_t MAX_LEVEL = 6;
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

class HNSWIndex : public VectorIndex {
private:
    struct Node {
    size_t id;
    std::vector<float> vector;
    std::vector<std::vector<size_t>> neighbors;
    size_t node_level;

    Node(size_t id, const std::vector<float>& vec, size_t node_level)
        : id(id), vector(vec), neighbors(MAX_LEVEL + 1), node_level(node_level) {}
};


    size_t max_level;         
    size_t M;                 // Max number of connections
    size_t M_max0;   
    float level_multiplier;   // To control level generation
    size_t ef_construction;   // Size of dynamic candidate list during construction
    size_t current_max_level;
    
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<size_t> entry_points; 
    size_t dimensions;
    BufferManager& buffer_manager;

    mutable std::vector<std::pair<float, size_t>> candidates_pool;
    mutable std::vector<bool> visited_pool;

public:
    HNSWIndex(size_t dims, BufferManager& bm) 
        : dimensions(dims), buffer_manager(bm), current_max_level(0) {
        setParameters(); 
    }
    
    void setParameters(size_t max_level_param = MAX_LEVEL,
                      size_t M_param = DEFAULT_M,
                      size_t ef_construction_param = DEFAULT_EF_CONSTRUCTION) {
        max_level = max_level_param;
        M = M_param;
        M_max0 = 2 * M;
        ef_construction = ef_construction_param;
        level_multiplier = 1.0f / std::log(M);
        
        entry_points.resize(max_level + 1, 0);

        candidates_pool.reserve(ef_construction * 2);
        visited_pool.resize(1000000, false);
    }
    
    // Random level generation based on the paper
    size_t getRandomLevel() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dis(0.0, 1.0);
        
        float r = -std::log(dis(gen)) * level_multiplier;
        return std::min(static_cast<size_t>(r), max_level);
    }
    
    float distance(const std::vector<float>& a, const std::vector<float>& b) const {
        return Field::computeL2Distance(a, b);
    }
    
    void selectNeighbors(const std::vector<float>& q,
                        const std::vector<std::pair<size_t, float>>& candidates,
                        size_t M_max,
                        std::vector<size_t>& selected) {
        
        selected.clear();
        if (candidates.empty()) return;
        
        std::vector<std::pair<size_t, float>> sorted_candidates = candidates;
        std::sort(sorted_candidates.begin(), sorted_candidates.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Take at most M_max neighbors
        for (size_t i = 0; i < std::min(M_max, sorted_candidates.size()); ++i) {
            selected.push_back(sorted_candidates[i].first);
        }
    }
    
    std::vector<std::pair<size_t, float>> searchLayer(
        const std::vector<float>& query,
        size_t ef,
        size_t ep,
        size_t layer) const {
        
        if (ep >= nodes.size()) return {};
        
        if (layer > nodes[ep]->node_level) return {};

        std::set<size_t> visited{ep};
        std::priority_queue<std::pair<float, size_t>> candidates;
        std::priority_queue<std::pair<float, size_t>> best;

        float dist_ep = distance(query, nodes[ep]->vector);
        candidates.push({-dist_ep, ep});
        best.push({dist_ep, ep});

        while (!candidates.empty()) {
            auto current = candidates.top();
            candidates.pop();

            if (-current.first > best.top().first) break;

            size_t current_id = current.second;
            
            if (layer <= nodes[current_id]->node_level) {
                for (size_t neighbor_id : nodes[current_id]->neighbors[layer]) {
                    if (neighbor_id >= nodes.size() || visited.count(neighbor_id) > 0) {
                        continue;
                    }
                    
                    visited.insert(neighbor_id);
                    float dist = distance(query, nodes[neighbor_id]->vector);
                    
                    if (best.size() < ef || dist < best.top().first) {
                        candidates.push({-dist, neighbor_id});
                        best.push({dist, neighbor_id});
                        
                        if (best.size() > ef) {
                            best.pop();
                        }
                    }
                }
            }
        }

        std::vector<std::pair<size_t, float>> results;
        while (!best.empty()) {
            results.push_back({best.top().second, best.top().first});
            best.pop();
        }
        
        std::reverse(results.begin(), results.end());
        return results;
    }
    
    void checkDimensions(const std::vector<float>& vector) const {
        ASSERT_WITH_MESSAGE(vector.size() == dimensions, 
            "Vector dimension mismatch: expected " + 
            std::to_string(dimensions) + ", got " + 
            std::to_string(vector.size()));
    }

    void checkNodeId(size_t id) const {
        ASSERT_WITH_MESSAGE(id < nodes.size(), 
            "Invalid node id: " + std::to_string(id) + 
            ", total nodes: " + std::to_string(nodes.size()));
    }

    void insert(const std::vector<float>& vector, size_t id) override {
        try {
            // std::cout << "[Insert] Starting insert for id " << id << "\n";
            checkDimensions(vector);
            
            size_t level = getRandomLevel();
            // std::cout << "[Insert] Generated level " << level << "\n";
            
            // If this is the first node
            if (nodes.empty()) {
                auto node = std::make_unique<Node>(id, vector, level);
                nodes.push_back(std::move(node));
                current_max_level = level;
                
                // Initialize entry points up to node's level
                for (size_t i = 0; i <= level; ++i) {
                    entry_points[i] = 0;
                }
                return;
            }

            size_t ep = entry_points[current_max_level];
            
            for (int lc = std::min(level, current_max_level); lc >= 0; --lc) {
                auto nearest = searchLayer(vector, 1, ep, lc);
                if (!nearest.empty()) {
                    ep = nearest[0].first;
                }
            }

            auto node = std::make_unique<Node>(id, vector, level);
            nodes.push_back(std::move(node));
            size_t node_idx = nodes.size() - 1;

            if (level > current_max_level) {
                for (size_t l = current_max_level + 1; l <= level; ++l) {
                    entry_points[l] = node_idx;
                }
                current_max_level = level;
            }

            for (size_t lc = 0; lc <= level; ++lc) {
                auto neighbors = searchLayer(vector, ef_construction, ep, lc);
                if (neighbors.empty()) continue;

                std::vector<size_t> selected;
                size_t M_max = (lc == 0) ? M_max0 : M;
                
                for (size_t i = 0; i < std::min(M_max, neighbors.size()); ++i) {
                    size_t neighbor_id = neighbors[i].first;
                    if (neighbor_id < nodes.size() && neighbor_id != node_idx) {
                        selected.push_back(neighbor_id);
                    }
                }

                for (size_t neighbor_id : selected) {
                    if (lc <= nodes[neighbor_id]->node_level) {
                        nodes[node_idx]->neighbors[lc].push_back(neighbor_id);
                        nodes[neighbor_id]->neighbors[lc].push_back(node_idx);
                    }
                }
            }
        } catch (const std::exception& e) {
            // std::cerr << "[Insert] Exception: " << e.what() << "\n";
            throw;
        }
    }

    std::vector<std::pair<size_t, float>> search(
        const std::vector<float>& query, 
        size_t k,
        size_t ef = 50) const override {
        
        // std::cout << "[Search] Starting with k=" << k 
        //           << ", ef=" << ef 
        //           << ", nodes.size=" << nodes.size() 
        //           << ", max_level=" << max_level << "\n";
        
        if (nodes.empty()) {
            // std::cout << "[Search] Index is empty\n";
            return {};
        }

        size_t ep = entry_points[max_level];
        // std::cout << "[Search] Initial entry point: " << ep << "\n";
        
        if (ep >= nodes.size()) {
            // std::cout << "[Search] Invalid entry point\n";
            return {};
        }

        for (int level = max_level; level >= 0; level--) {
            // std::cout << "[Search] Searching layer " << level 
                    //   << " with entry point " << ep << "\n";
            
            if (level > 0) {
                auto layer_results = searchLayer(query, 1, ep, level);
                if (!layer_results.empty()) {
                    ep = layer_results[0].first;
                    // std::cout << "[Search] New entry point: " << ep << "\n";
                } else {
                    // std::cout << "[Search] No results found at layer " << level << "\n";
                }
            } else {
                // std::cout << "[Search] Searching ground layer with ef=" 
                        //   << std::max(ef, k) << "\n";
                auto results = searchLayer(query, std::max(ef, k), ep, 0);
                
                // std::cout << "[Search] Found " << results.size() 
                        //   << " results at ground layer\n";
                
                if (results.size() > k) {
                    results.resize(k);
                }
                return results;
            }
        }
        
        return {};
    }
};

class KDTreeIndex : public VectorIndex {
private:
    struct Node {
        std::vector<float> point;
        size_t point_id;
        size_t split_dim;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        
        Node(const std::vector<float>& p, size_t id) 
            : point(p), point_id(id), split_dim(0) {}
    };
    
    std::unique_ptr<Node> root;
    size_t dimensions;
    BufferManager& buffer_manager;
    
    Node* insertRecursive(std::unique_ptr<Node>& node, 
                         const std::vector<float>& point,
                         size_t point_id,
                         size_t depth) {
        if (!node) {
            node = std::make_unique<Node>(point, point_id);
            node->split_dim = depth % dimensions;
            return node.get();
        }
        
        size_t dim = depth % dimensions;
        if (point[dim] < node->point[dim]) {
            return insertRecursive(node->left, point, point_id, depth + 1);
        } else {
            return insertRecursive(node->right, point, point_id, depth + 1);
        }
    }
    
    void searchKNN(Node* node,
                   const std::vector<float>& query,
                   std::priority_queue<std::pair<float, size_t>>& pq,
                   size_t k,
                   size_t depth) const {
        if (!node) return;
        
        float dist = Field::computeL2Distance(query, node->point);
        
        if (pq.size() < k) {
            pq.push({dist, node->point_id});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, node->point_id});
        }
        
        size_t dim = depth % dimensions;
        float diff = query[dim] - node->point[dim];
        
        if (diff < 0) {
            searchKNN(node->left.get(), query, pq, k, depth + 1);
            if (pq.size() < k || std::abs(diff) < pq.top().first) {
                searchKNN(node->right.get(), query, pq, k, depth + 1);
            }
        } else {
            searchKNN(node->right.get(), query, pq, k, depth + 1);
            if (pq.size() < k || std::abs(diff) < pq.top().first) {
                searchKNN(node->left.get(), query, pq, k, depth + 1);
            }
        }
    }
    
    float computeDistance(const std::vector<float>& a, 
                         const std::vector<float>& b) const {
        float dist = 0.0f;
        for (size_t i = 0; i < dimensions; i++) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
    
public:
    KDTreeIndex(size_t dims, BufferManager& bm) 
        : dimensions(dims), buffer_manager(bm) {}
    
    void insert(const std::vector<float>& point, size_t point_id) override {
        insertRecursive(root, point, point_id, 0);
    }
    
    std::vector<std::pair<size_t, float>> search(
        const std::vector<float>& query,
        size_t k,
        size_t ef = 50) const override {
        
        std::priority_queue<std::pair<float, size_t>> pq;
        searchKNN(root.get(), query, pq, k, 0);
        
        std::vector<std::pair<size_t, float>> results;
        while (!pq.empty()) {
            results.push_back({pq.top().second, pq.top().first});
            pq.pop();
        }
        std::reverse(results.begin(), results.end());
        return results;
    }
};

class VectorIndexBenchmark {
private:
    BufferManager buffer_manager;

public:
    VectorIndexBenchmark() : buffer_manager(true) {}

    struct Metrics {
        double build_time;
        double query_time;
        double memory_usage;
        double accuracy;
    };

    struct IndexConfig {
        std::string index_type;
        size_t max_level;
        size_t neighbors_size;
        size_t ef_construction;
        size_t leaf_size;
    };

    Metrics runBenchmark(const IndexConfig& config,
                        const std::vector<std::vector<float>>& data,
                        const std::vector<std::vector<float>>& queries,
                        size_t k) {
        Metrics metrics;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::unique_ptr<VectorIndex> index;
        if (config.index_type == "kdtree") {
            index = std::make_unique<KDTreeIndex>(data[0].size(), buffer_manager);
        } else if (config.index_type == "hnsw") {
            auto hnsw = std::make_unique<HNSWIndex>(data[0].size(), buffer_manager);
            hnsw->setParameters(config.max_level, config.neighbors_size, 
                              config.ef_construction);
            index = std::move(hnsw);
        }

        for (size_t i = 0; i < data.size(); ++i) {
            index->insert(data[i], i);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        metrics.build_time = std::chrono::duration<double>(end - start).count();

        // Measure query time
        start = std::chrono::high_resolution_clock::now();
        for (const auto& query : queries) {
            index->search(query, k);
        }
        end = std::chrono::high_resolution_clock::now();
        metrics.query_time = std::chrono::duration<double>(end - start).count() 
                           / queries.size();

        // Measure memory usage
        metrics.memory_usage = getCurrentMemoryUsage();

        // Measure accuracy
        metrics.accuracy = measureAccuracy(index.get(), queries, data, k);

        return metrics;
    }

private:
    std::vector<std::pair<size_t, float>> linearSearch(
        const std::vector<float>& query,
        const std::vector<std::vector<float>>& data,
        size_t k) {
        std::priority_queue<std::pair<float, size_t>> pq;
        
        for (size_t i = 0; i < data.size(); i++) {
            float dist = Field::computeL2Distance(query, data[i]);
            if (pq.size() < k) {
                pq.push({dist, i});
            } else if (dist < pq.top().first) {
                pq.pop();
                pq.push({dist, i});
            }
        }
        
        std::vector<std::pair<size_t, float>> result;
        while (!pq.empty()) {
            result.push_back({pq.top().second, pq.top().first});
            pq.pop();
        }
        std::reverse(result.begin(), result.end());
        return result;
    }

    double measureAccuracy(VectorIndex* index,
                          const std::vector<std::vector<float>>& queries,
                          const std::vector<std::vector<float>>& data,
                          size_t k) {
        double total_accuracy = 0.0;
        
        for (const auto& query : queries) {
            auto approx_results = index->search(query, k);
            auto exact_results = linearSearch(query, data, k);
            
            size_t correct = 0;
            for (const auto& res : approx_results) {
                if (std::find_if(exact_results.begin(), exact_results.end(),
                    [&](const auto& exact) { 
                        return exact.first == res.first; 
                    }) != exact_results.end()) {
                    correct++;
                }
            }
            total_accuracy += static_cast<double>(correct) / k;
        }
        
        return total_accuracy / queries.size();
    }
};

class IndexAutoTuner {
private:
    struct TuningResult {
        VectorIndexBenchmark::IndexConfig config;
        VectorIndexBenchmark::Metrics metrics;
        double score;
    };

public:
    VectorIndexBenchmark::IndexConfig tune(
        const std::vector<std::vector<float>>& sample_data,
        const std::vector<std::vector<float>>& sample_queries,
        size_t k,
        double accuracy_weight = 0.4,
        double speed_weight = 0.4,
        double memory_weight = 0.2) {
        
        std::vector<TuningResult> results;
        VectorIndexBenchmark benchmark;


        std::vector<VectorIndexBenchmark::IndexConfig> configs = {

            {"kdtree", 0, 0, 0, 10},
            {"kdtree", 0, 0, 0, 20},
            {"kdtree", 0, 0, 0, 50},
            

            {"hnsw", 4, 16, 100, 0},
            {"hnsw", 4, 32, 200, 0},
            {"hnsw", 6, 16, 100, 0},
            {"hnsw", 6, 32, 200, 0}
        };


        for (const auto& config : configs) {
            auto metrics = benchmark.runBenchmark(config, sample_data, 
                                                sample_queries, k);
            
            double score = 
                accuracy_weight * metrics.accuracy +
                speed_weight * (1.0 / metrics.query_time) +
                memory_weight * (1.0 / metrics.memory_usage);
            
            results.push_back({config, metrics, score});
        }

        auto best = std::max_element(results.begin(), results.end(),
            [](const TuningResult& a, const TuningResult& b) {
                return a.score < b.score;
            });

        return best->config;
    }
};

class VectorQueryExecutor {
private:
    BufferManager& buffer_manager;
    std::unique_ptr<HNSWIndex> hnsw_index;
    std::unique_ptr<KDTreeIndex> kd_index;
    
public:
    enum class QueryType {
        KNN_SEARCH,
        RANGE_SEARCH,
        BATCH_SEARCH
    };
    
    struct QueryResult {
        std::vector<std::pair<size_t, float>> matches;
        double execution_time;
        std::string index_used;
    };
    
    VectorQueryExecutor(BufferManager& bm) 
        : buffer_manager(bm) {
        hnsw_index = std::make_unique<HNSWIndex>(VECTOR_DIMENSION, buffer_manager);
        kd_index = std::make_unique<KDTreeIndex>(VECTOR_DIMENSION, buffer_manager);
        
        hnsw_index->setParameters(4, 16, 100);
    }
    
    QueryResult executeQuery(const std::vector<float>& query_vector,
                           QueryType query_type,
                           size_t k = 10,
                           float radius = 1.0f) {
        QueryResult result;
        auto start = std::chrono::high_resolution_clock::now();
        
        bool use_hnsw = shouldUseHNSW(query_type, k);
        
        switch (query_type) {
            case QueryType::KNN_SEARCH:
                if (use_hnsw) {
                    result.matches = hnsw_index->search(query_vector, k);
                    result.index_used = "HNSW";
                } else {
                    result.matches = kd_index->search(query_vector, k);
                    result.index_used = "KD-tree";
                }
                break;
                
            case QueryType::RANGE_SEARCH:
                result.matches = kd_index->rangeSearch(query_vector, radius);
                result.index_used = "KD-tree";
                break;
                
            case QueryType::BATCH_SEARCH:
                result.matches = executeBatchSearch(query_vector, k);
                result.index_used = "Batch-optimized";
                break;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.execution_time = 
            std::chrono::duration<double>(end - start).count();
        
        return result;
    }
    
    void insertVector(const std::vector<float>& vector, size_t id) {
        hnsw_index->insert(vector, id);
        kd_index->insert(vector, id);
    }
    
private:
    bool shouldUseHNSW(QueryType query_type, size_t k) {
        if (query_type == QueryType::RANGE_SEARCH) {
            return false;
        }
        if (k > 100) {
            return false;
        }
        return true;
    }
    
    std::vector<std::pair<size_t, float>> executeBatchSearch(
        const std::vector<float>& query,
        size_t k) {
        return hnsw_index->search(query, k);
    }
};

void testHNSW() {
    std::cout << "Starting HNSW test...\n";
    
    BufferManager bm(true);
    std::cout << "Created BufferManager\n";
    
    HNSWIndex index(3, bm);
    std::cout << "Created HNSW index\n";
    
    std::vector<std::vector<float>> points = {
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f}
    };
    std::cout << "Created test points\n";
    
    for (size_t i = 0; i < points.size(); i++) {
        std::cout << "Inserting point " << i << "\n";
        index.insert(points[i], i);
        std::cout << "Inserted point " << i << "\n";
    }
    
    std::cout << "All points inserted\n";
    
    std::cout << "Starting search...\n";
    auto results = index.search(points[0], 1);
    std::cout << "Search completed\n";
    
    ASSERT_WITH_MESSAGE(results.size() == 1, "Should find exactly one match");
    ASSERT_WITH_MESSAGE(results[0].first == 0, "Should find the exact point");
    
    std::cout << "HNSW tests passed!\n";
}

void testFieldSerialization() {
    std::cout << "Running testFieldSerialization()...\n";
    const size_t dim = 128;
    std::vector<float> original(dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto &val : original) {
        val = dist(gen);
    }

    Field originalField(original);
    
    std::stringstream ss;
    ss << originalField.serialize();

    std::stringstream iss(ss.str());
    auto deserializedField = Field::deserialize(iss);
    ASSERT_WITH_MESSAGE(deserializedField != nullptr, "Deserialization failed");

    auto recovered = deserializedField->asVector();
    ASSERT_WITH_MESSAGE(recovered.size() == original.size(), "Dimension mismatch after deserialization");
    for (size_t i = 0; i < dim; i++) {
        ASSERT_WITH_MESSAGE(std::fabs(recovered[i] - original[i]) < 1e-6f,
                            "Value mismatch in deserialized vector");
    }

    std::cout << "testFieldSerialization passed!\n";
}

double computeRecall(
    const std::vector<std::pair<size_t, float>>& approx_results,
    const std::vector<std::pair<size_t, float>>& exact_results) 
{
    size_t correct = 0;
    for (auto &res : approx_results) {
        for (auto &ex : exact_results) {
            if (res.first == ex.first) {
                correct++;
                break;
            }
        }
    }
    return static_cast<double>(correct) / exact_results.size();
}

void testKNNAccuracy(size_t dataset_size = 1000, size_t dims = 64, size_t k = 10) {
    std::cout << "Running testKNNAccuracy() with dataset_size=" << dataset_size 
              << ", dims=" << dims << ", k=" << k << "\n";

    std::vector<std::vector<float>> data;
    data.reserve(dataset_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < dataset_size; i++) {
        std::vector<float> vec(dims);
        for (auto &val : vec) val = dist(gen);
        data.push_back(std::move(vec));
    }

    BufferManager bm(true);
    KDTreeIndex kd(dims, bm);
    HNSWIndex hnsw(dims, bm);
    hnsw.setParameters(4,16,100);
    
    for (size_t i = 0; i < dataset_size; i++) {
        kd.insert(data[i], i);
        hnsw.insert(data[i], i);
    }

    size_t test_queries = 100;
    double total_recall = 0.0;
    for (size_t qi = 0; qi < test_queries; qi++) {
        size_t query_id = qi % dataset_size;
        auto exact = kd.search(data[query_id], k);
        auto approx = hnsw.search(data[query_id], k);

        double recall = computeRecall(approx, exact);
        total_recall += recall;
    }

    double avg_recall = total_recall / test_queries;
    std::cout << "Average recall over " << test_queries << " queries: " << (avg_recall*100) << "%\n";
    ASSERT_WITH_MESSAGE(avg_recall > 0.8, "Recall is too low, expected at least 80%");
    std::cout << "testKNNAccuracy passed!\n";
}

void testBufferManagerLargeVectors() {
    std::cout << "Running testBufferManagerLargeVectors()...\n";

    size_t large_dim = 1000; 
    std::vector<float> large_vec(large_dim, 42.0f);
    auto tuple = std::make_unique<Tuple>();
    tuple->addField(std::make_unique<Field>(large_vec));

    BufferManager bm(true);
    bm.extend();
    for (int i = 0; i < 5; i++) {
        auto &page = bm.fix_page(i);
        bool success = page.addTuple(tuple->clone());

        // If doesn't fit, try next page
        if (!success) {
            bm.extend();
        }
        bm.flushPage(i);
    }

    std::cout << "testBufferManagerLargeVectors passed!\n";
}

void testMemoryUsage() {
    std::cout << "Running testMemoryUsage()...\n";
    BufferManager bm(true);
    KDTreeIndex kd(128, bm);
    
    std::vector<std::vector<float>> data;
    size_t data_size = 5000;
    data.reserve(data_size);
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < data_size; i++) {
        std::vector<float> v(128);
        for (auto &val : v) val = dist(gen);
        data.push_back(std::move(v));
    }
    
    double mem_before = 0.0;
    double mem_after = 0.0;
    mem_before = getCurrentMemoryUsage(); 
    for (size_t i = 0; i < data_size; i++) {
        kd.insert(data[i], i);
    }
    mem_after = getCurrentMemoryUsage();

    std::cout << "Memory before: " << mem_before << " bytes\n"
              << "Memory after: " << mem_after << " bytes\n";
    
    // Temporary check
    ASSERT_WITH_MESSAGE(mem_after >= mem_before, "Memory usage after building should be >= before (expected).");
    std::cout << "testMemoryUsage passed!\n";
}

std::vector<std::vector<float>> loadGIST1M(const std::string &filename, size_t count) {
    std::ifstream in(filename, std::ios::binary);
    if(!in) {
        throw std::runtime_error("Could not open GIST1M file");
    }

    std::vector<std::vector<float>> data;
    data.reserve(count);

    int dim;
    for (size_t i = 0; i < count; i++) {
        in.read(reinterpret_cast<char*>(&dim), 4);
        std::vector<float> vec(dim);
        in.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if(!in) break;
        data.push_back(std::move(vec));
    }
    return data;
}

void benchmarkGIST1M() {
    std::cout << "Running benchmarkGIST1M...\n";
    size_t subset_size = 10000; 
    auto data = loadGIST1M("gist1M.fvecs", subset_size);

    std::vector<std::vector<float>> queries(data.begin(), data.begin() + 100);

    VectorIndexBenchmark::IndexConfig hnsw_config{"hnsw", 4,16,100,0};
    VectorIndexBenchmark benchmark;
    auto metrics = benchmark.runBenchmark(hnsw_config, data, queries, 10);

    std::cout << "GIST1M Results:\n"
              << "Build time: " << metrics.build_time << "s\n"
              << "Query time: " << metrics.query_time << "s\n"
              << "Memory usage: " << metrics.memory_usage / 1024 / 1024 << "MB\n"
              << "Accuracy: " << metrics.accuracy * 100 << "%\n";

    ASSERT_WITH_MESSAGE(metrics.accuracy > 0.9, "Expected >90% accuracy on GIST1M subset");
    std::cout << "benchmarkGIST1M passed!\n";
}

void tuneHNSWParameters(std::vector<std::vector<float>>& data,
                        std::vector<std::vector<float>>& queries,
                        size_t k) {
    std::vector<size_t> Ms = {8,16,32};
    std::vector<size_t> efCs = {100,200,300};
    std::vector<size_t> levels = {4,6};

    VectorIndexBenchmark benchmark;
    double best_score = -1;
    VectorIndexBenchmark::IndexConfig best_config;

    for (auto M : Ms) {
        for (auto efC : efCs) {
            for (auto lvl : levels) {
                VectorIndexBenchmark::IndexConfig config{"hnsw", lvl, M, efC, 0};
                auto metrics = benchmark.runBenchmark(config, data, queries, k);

                double score = metrics.accuracy * (1.0/metrics.query_time);
                if (score > best_score) {
                    best_score = score;
                    best_config = config;
                }
            }
        }
    }

    std::cout << "Best config: (level=" << best_config.max_level 
              << ", M=" << best_config.neighbors_size 
              << ", efC=" << best_config.ef_construction 
              << ") with score=" << best_score << "\n";
}

void compareAndLogResults(
    VectorIndex* hnsw, 
    VectorIndex* kd, 
    const std::vector<std::vector<float>>& queries, 
    size_t k) 
{
    double total_recall = 0.0;
    for (size_t i = 0; i < queries.size(); i++) {
        auto exact = kd->search(queries[i], k);
        auto approx = hnsw->search(queries[i], k);
        double recall = computeRecall(approx, exact);
        total_recall += recall;
    }
    double avg_recall = total_recall / queries.size();
    std::cout << "[CompareAndLog] Average recall over " << queries.size() 
              << " queries: " << (avg_recall*100) << "%\n";
}


#ifndef BUZZDB_MAIN_DISABLED
int main() {
    try {
        std::cout << "Starting Vector Database Demo...\n\n";
        
        // Run tests
        testHNSW();
        testFieldSerialization();
        testKNNAccuracy(1000, 64, 10);
        testBufferManagerLargeVectors();
        testMemoryUsage();
        
        std::cout << "\nAll tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
#endif