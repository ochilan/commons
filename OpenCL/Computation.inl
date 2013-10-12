/*
Computation::Ptr Computation::fromSourceFiles(const std::vector<std::string>& paths) {
	Computation::Ptr self(new Computation());

	cl_int err;

	// platform
	std::vector<cl::Platform> platforms;
	err = cl::Platform::get(&platforms);
	if (!checkGetPlatform(err)) return Ptr();

	// device
	self->m_devices = DevicesPtr(new Devices());
	err = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, self->m_devices.get());
	if (!checkGetDevices(err, self->m_devices->size())) return Ptr();
	
	// context
	self->m_context = ContextPtr(new cl::Context(*(self->m_devices), NULL, NULL, NULL, &err));
	if (!checkGetContext(err)) return Ptr();

	// load sources
	if (!paths.size()) {
		Log::error("Empty source path set given.");
		return Ptr();
	}
	std::vector<std::string> codes;
	cl::Program::Sources sources;
	std::ifstream in;
	for (const auto& p : paths) {
		in.open(p.c_str());
		if (!in.good()) {
			Log::error("Could not load file \""+p+"\".");
			return Ptr();
		}
		std::istreambuf_iterator<char> begin(in), end;
		std::string code(begin, end);
		in.close();
		codes.push_back(code);
		sources.push_back({code.c_str(), code.length()+1});
	}

	// program
	self->m_program = ProgramPtr(new cl::Program(*(self->m_context), sources, &err));
	if (!checkCreateProgram(err))	return Ptr();

	err = self->m_program->build(*(self->m_devices));
	if (!checkBuildProgram(err, self->m_devices, self->m_program)) return Ptr();

	self->m_queue = QueuePtr(new cl::CommandQueue(*(self->m_context), (*(self->m_devices))[0], 0, &err));
	if (!checkCreateQueue(err)) return Ptr();

	self->m_allocated = std::shared_ptr<unsigned long>(new unsigned long(0));

	return self;
}
*/

inline Computation::ContextPtr Computation::getContext() {
	return m_context;
}

inline Computation::ProgramPtr Computation::getProgram() {
	return m_program;
}

inline Computation::DevicesPtr Computation::getDevices() {
	return m_devices;
}

inline Computation::QueuePtr Computation::getQueue() {
	return m_queue;
}

inline Computation::FunctorPtr Computation::getFunctor(std::string functionName, const cl::NDRange& global, const cl::NDRange& local, const cl::NDRange& offset) {
	cl_int err;
	cl::Kernel kernel(*m_program, functionName.c_str(), &err);
	if (!checkGetFunctor(err)) return FunctorPtr();

	FunctorPtr functor(new cl::KernelFunctor(kernel, *m_queue, offset, global, local));
	return functor;
}

inline cl_ulong Computation::memInBytesGlobal() const {
	cl_ulong result;
	(*m_devices)[0].getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &result);
	return result;
}

inline cl_ulong Computation::memInBytesPerAlloc() const {
	cl_ulong result;
	(*m_devices)[0].getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &result);
	return result;
}

inline cl_ulong Computation::memInBytesAvailable() const {
	return std::min(memInBytesPerAlloc(), memInBytesGlobal()- *m_allocated);
}

inline double Computation::memInKBytesGlobal() const {
	return static_cast<double>(memInBytesGlobal()) / 1024.f;
}

inline double Computation::memInKBytesPerAlloc() const {
	return static_cast<double>(memInBytesPerAlloc()) / 1024.f;
}

inline double Computation::memInKBytesAvailable() const {
	return static_cast<double>(memInBytesAvailable()) / 1024.f;
}

inline double Computation::memInMBytesGlobal() const {
	return static_cast<double>(memInBytesGlobal()) / (1024.f * 1024.f);
}

inline double Computation::memInMBytesPerAlloc() const {
	return static_cast<double>(memInBytesPerAlloc()) / (1024.f * 1024.f);
}

inline double Computation::memInMBytesAvailable() const {
	return static_cast<double>(memInBytesAvailable()) / (1024.f * 1024.f);
}

template <class T>
inline unsigned long Computation::numValuesGlobal() const {
	return memInBytesGlobal() / sizeof(T);
}

template <class T>
inline unsigned long Computation::numValuesPerAlloc() const {
	return memInBytesPerAlloc() / sizeof(T);
}

template <class T>
inline unsigned long Computation::numValuesAvailable() const {
	return memInBytesAvailable() / sizeof(T);
}


template <class... Args>
bool Computation::run(FunctorPtr functor, Args... args) {
	(*functor)(args...);
		
	cl_int err = functor->getError();
	return checkRunKernel(err);
}

template <int blockSize, class Scalar, class... Args>
bool Computation::runBlock1D(unsigned int dim, Eigen::VectorXf& result, std::string funcName, Args... args) {
	auto functor = getFunctor(funcName, cl::NDRange(blockSize));
	if (!functor) return false;

	result.resize(dim);
	auto sub = getMatrixBuffer<Scalar>(blockSize, 1, CL_MEM_WRITE_ONLY);
	auto params = getMatrixBuffer<unsigned int>(3, 1, CL_MEM_READ_ONLY);
	params->data()[0] = dim;
	unsigned int* begin = &(params->data()[1]);
	unsigned int*   end = &(params->data()[2]);
	for (*begin = 0; *begin < dim; *begin += blockSize) {
		*end = std::min(*begin+blockSize, dim);
		params->write();
		// compute submatrix
		if (!run(functor, params->buffer(), sub->buffer(), args...)) return false;
		sub->read();
		result.segment(*begin, *end - *begin) = sub->eigenMap().block(0, 0, *end - *begin, 1);
	}

	return true;
}

template <class Scalar, class... Args>
bool Computation::runBlock1DDyn(unsigned int dim, Eigen::VectorXf& result, std::string funcName, Args... args) {
	int blockSize = numValuesAvailable<Scalar>();

	auto functor = getFunctor(funcName, cl::NDRange(blockSize));
	if (!functor) return false;

	result.resize(dim);
	auto sub = getMatrixBuffer<Scalar>(blockSize, 1, CL_MEM_WRITE_ONLY);
	auto params = getMatrixBuffer<unsigned int>(3, 1, CL_MEM_READ_ONLY);
	params->data()[0] = dim;
	unsigned int* begin = &(params->data()[1]);
	unsigned int*   end = &(params->data()[2]);
	for (*begin = 0; *begin < dim; *begin += blockSize) {
		*end = std::min(*begin+blockSize, dim);
		params->write();
		// compute submatrix
		if (!run(functor, params->buffer(), sub->buffer(), args...)) return false;
		sub->read();
		result.segment(*begin, *end - *begin) = sub->eigenMap().block(0, 0, *end - *begin, 1);
	}

	return true;
}

template <int rowCount, class Scalar, class... Args>
bool Computation::runBlock2D(unsigned int dim, Eigen::MatrixXf& result, std::string funcName, Args... args) {
	auto functor = getFunctor(funcName, cl::NDRange(rowCount, dim));
	if (!functor) return false;

	result.resize(dim, dim);
	auto sub = getMatrixBuffer<Scalar>(rowCount, dim, CL_MEM_WRITE_ONLY);
	auto params = getMatrixBuffer<unsigned int>(3, 1, CL_MEM_READ_ONLY);
	params->data()[0] = dim;
	unsigned int* begin = &(params->data()[1]);
	unsigned int*   end = &(params->data()[2]);
	for (*begin = 0; *begin < dim; *begin += rowCount) {
		*end = std::min(*begin+rowCount, dim);
		params->write();
		// compute submatrix
		if (!run(functor, params->buffer(), sub->buffer(), args...)) return false;
		sub->read();
		result.block(*begin, 0, *end-*begin, dim) = sub->eigenMap().block(0, 0, *end - *begin, dim);
	}

	return true;
}

template <class Scalar, class... Args>
bool Computation::runBlock2DDyn(unsigned int dim, Eigen::MatrixXf& result, std::string funcName, Args... args) {
	// determine rowCount
	int rowCount = (numValuesAvailable<float>() / dim) - 1;

	auto functor = getFunctor(funcName, cl::NDRange(rowCount, dim));
	if (!functor) return false;

	result.resize(dim, dim);
	auto sub = getMatrixBuffer<Scalar>(rowCount, dim, CL_MEM_WRITE_ONLY);
	auto params = getMatrixBuffer<unsigned int>(3, 1, CL_MEM_READ_ONLY);
	params->data()[0] = dim;
	unsigned int* begin = &(params->data()[1]);
	unsigned int*   end = &(params->data()[2]);
	for (*begin = 0; *begin < dim; *begin += rowCount) {
		*end = std::min(*begin+rowCount, dim);
		params->write();
		// compute submatrix
		if (!run(functor, params->buffer(), sub->buffer(), args...)) return false;
		sub->read();
		result.block(*begin, 0, *end-*begin, dim) = sub->eigenMap().block(0, 0, *end - *begin, dim);
	}

	return true;
}

#ifdef OPENCL_USE_EIGEN
template <class Scalar, int Dim>
inline typename MatrixBuffer<Scalar>::Ptr Computation::getMatrixBuffer(const std::vector<Eigen::Matrix<Scalar, Dim, 1>>& vectors, cl_int mode) {
	checkAvailableSpace<Scalar>(Dim*vectors.size());
	typename MatrixBuffer<Scalar>::Ptr result(new MatrixBuffer<Scalar>(m_queue, vectors, *m_context, mode, m_allocated));
	return result;
}
template <class Scalar>
inline typename MatrixBuffer<Scalar>::Ptr Computation::getMatrixBuffer(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& matrix, cl_int mode) {
	checkAvailableSpace<Scalar>(matrix.rows()*matrix.cols());
	typename MatrixBuffer<Scalar>::Ptr result(new MatrixBuffer<Scalar>(m_queue, matrix, *m_context, mode, m_allocated));
	return result;
}
#endif

#ifdef OPENCL_USE_PCL
inline typename MatrixBuffer<cl_float>::Ptr Computation::getMatrixBuffer(const pcl::PointCloud<pcl::PointNormal>& cloud, cl_int mode, const std::vector<int>& subset) {
	checkAvailableSpace<cl_float>(6*cloud.size());
	typename MatrixBuffer<cl_float>::Ptr result(new MatrixBuffer<cl_float>(m_queue, cloud, *m_context, mode, subset, m_allocated));
	return result;
}
#endif

template <class Scalar>
inline typename MatrixBuffer<Scalar>::Ptr Computation::getMatrixBuffer(int rows, int cols, cl_int mode) {
	checkAvailableSpace<Scalar>(static_cast<unsigned long>(rows*cols));
	typename MatrixBuffer<Scalar>::Ptr result(new MatrixBuffer<Scalar>(m_queue, rows, cols, *m_context, mode, m_allocated));
	return result;
}

template <class Scalar>
inline void Computation::checkAvailableSpace(unsigned long count) {
	unsigned long available = memInBytesAvailable();
	unsigned long requested = sizeof(Scalar)*count;
	if (requested > available) throw Error::OutOfMemAlloc(static_cast<double>(requested)/(1024.f*1024.f), static_cast<float>(available)/(1024.f*1024.f));
}

inline bool Computation::checkGetPlatform(cl_int err) {
	if (err != CL_SUCCESS) {
		Log::error("Unable to get GPU Devices: No platform");
		return false;
	}
	return true;
}

inline bool Computation::checkGetDevices(cl_int err, unsigned int deviceCount) {
	if (err != CL_SUCCESS || !deviceCount) {
		Log::error("Unable to get GPU Devices: No devices for this platform");
		return false;
	}
	return true;
}

inline bool Computation::checkGetContext(cl_int err) {
	if (err != CL_SUCCESS) {
		Log::error("Unable to get OpenCL context. Reason:");
		switch (err) {
			case CL_INVALID_DEVICE: Log::error("Invalid Device"); break;
			case CL_DEVICE_NOT_AVAILABLE: Log::error("Device Not Available"); break;
			case CL_OUT_OF_HOST_MEMORY: Log::error("Out Of Host Memory"); break;
			default: Log::error("Unknown reason...");
		}
		return false;
	}
	return true;
}

inline bool Computation::checkCreateProgram(cl_int err) {
	if (err != CL_SUCCESS) {
		Log::error("Could not load program. Reason:");
		switch (err) {
			case CL_INVALID_CONTEXT: Log::error("Invalid Context"); break;
			case CL_INVALID_VALUE: Log::error("Invalid Value"); break;
			case CL_OUT_OF_RESOURCES: Log::error("Out Of Resources"); break;
			case CL_OUT_OF_HOST_MEMORY: Log::error("Out Of Host Memory"); break;;
			default: Log::error("Unknown reason...");
		}
		return false;
	}
	return true;
}

inline bool Computation::checkBuildProgram(cl_int err, DevicesPtr devices, ProgramPtr program) {
	if (err != CL_SUCCESS) {
		Log::error("Could not compile program. Reason:");
		switch (err) {
			case CL_INVALID_VALUE: Log::error("Invalid Value"); break;
			case CL_INVALID_DEVICE: Log::error("Invalid Device"); break;
			case CL_INVALID_BINARY: Log::error("Invalid Binary"); break;
			case CL_INVALID_BUILD_OPTIONS: Log::error("Invalid Build Options"); break;
			case CL_INVALID_OPERATION: Log::error("Invalid Operation"); break;
			case CL_COMPILER_NOT_AVAILABLE: Log::error("Compiler Not Available"); break;
			case CL_BUILD_PROGRAM_FAILURE: Log::error("Build Program Failure"); break;
			default: Log::error("Unknown reason..."); break;
		}
		if (err == CL_BUILD_PROGRAM_FAILURE) {
			for (auto& device : *devices) {
				std::string log;
				program->getBuildInfo(device, CL_PROGRAM_BUILD_LOG, &log);
				if (!log.empty()) {
					Log::error(log);
				}
			}
		}
		return false;
	}
	return true;
}

inline bool Computation::checkCreateQueue(cl_int err) {
	if (err != CL_SUCCESS) {
		Log::error("Could not create command queue");
		return false;
	}
	return true;
}

inline bool Computation::checkRunKernel(cl_int err) {
	if (err != CL_SUCCESS) {
		Log::error("Could not run kernel. Reason:");
		switch (err) {
			case CL_INVALID_PROGRAM_EXECUTABLE: Log::error("Invalid Program Executable"); break;
			case CL_INVALID_KERNEL: Log::error("Invalid Kernel"); break;
			case CL_INVALID_CONTEXT: Log::error("Invalid Context"); break;
			case CL_INVALID_KERNEL_ARGS: Log::error("Invalid Kernel Args"); break;
			case CL_INVALID_GLOBAL_WORK_SIZE: Log::error("Invalid Global Work Size"); break;
			case CL_INVALID_GLOBAL_OFFSET: Log::error("Invalid Global Offset"); break;
			case CL_INVALID_WORK_GROUP_SIZE: Log::error("Invalid Work Group Size"); break;
			case CL_DEVICE_MAX_WORK_GROUP_SIZE: Log::error("Device Max Work Group Size"); break;
			case CL_INVALID_WORK_ITEM_SIZE: Log::error("Invalid Work Item Size"); break;
			case CL_MISALIGNED_SUB_BUFFER_OFFSET: Log::error("Misaligned Sub Buffer Offset"); break;
			case CL_INVALID_IMAGE_SIZE: Log::error("Invalid Image Size"); break;
			case CL_OUT_OF_RESOURCES: Log::error("Out Of Resources"); break;
			case CL_MEM_OBJECT_ALLOCATION_FAILURE: Log::error("Mem Object Allocation Failure"); break;
			case CL_INVALID_EVENT_WAIT_LIST: Log::error("Invalid Event Wait List"); break;
			default: Log::error("Unknown error"); break;
		}
		return false;
	}
	return true;
}

inline bool Computation::checkGetFunctor(cl_int err) {
	if (err != CL_SUCCESS) {
		Log::error("Could not load kernel function. Reason:");
		switch (err) {
			case CL_INVALID_PROGRAM: Log::error("Invalid Program"); break;
			case CL_INVALID_PROGRAM_EXECUTABLE: Log::error("Invalid Program Executable"); break;
			case CL_INVALID_KERNEL_NAME: Log::error("Invalid Kernel Name"); break;
			case CL_INVALID_KERNEL_DEFINITION: Log::error("Invalid Kernel Definition"); break;
			case CL_INVALID_VALUE: Log::error("Invalid Value"); break;
			case CL_OUT_OF_RESOURCES: Log::error("Out Of Resources"); break;
			case CL_OUT_OF_HOST_MEMORY: Log::error("Out Of Host Memory"); break;
			default: Log::error("Unknown reason..."); break;
		}
		return false;
	}
	return true;
}
