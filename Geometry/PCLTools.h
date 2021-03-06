#ifndef COMMONSPCLTOOLS_H_
#define COMMONSPCLTOOLS_H_

#include <vector>
#include <memory>

#include <boost/variant.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/pcl_search.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/principal_curvatures.h>

#include <Eigen/Dense>
using Eigen::Vector3f;
using Eigen::Vector4f;
using Eigen::Matrix3f;
using Eigen::Matrix4f;

#include <Algorithm/Sets.h>
#include <IO/AbstractProgressBar.h>

#include "Quadric.h"

namespace Geometry {

template <class PointT>
struct NeighborQuery {
	typename pcl::search::Search<PointT>::Ptr   search; // must always have target point cloud set as InputCloud
	boost::variant<float,unsigned int>          param;  // type determines whether radius- or knn-search is used
};

template <class PointT>
class PCLTools {
	public:
		typedef std::shared_ptr<PCLTools> Ptr;
		typedef std::weak_ptr<PCLTools>   WPtr;
		
		typedef pcl::PointCloud<PointT>      CloudType;
		typedef pcl::search::Search<PointT>  SearchType;
		typedef pcl::search::KdTree<PointT>  KdTreeType;
		typedef NeighborQuery<PointT>        NQ;
		typedef Quadric<float, 2>            QuadricType;

		typedef std::vector<PointT>          Points;

		typedef int                          Idx;
		typedef int                          Size;
		typedef std::vector<int>             IdxSet;

		typedef IO::AbstractProgressBar      PBar;

	public:
		PCLTools() = delete;

		//static typename CloudType::Ptr loadPointCloud(fs::path cloudPath, std::string upAxis, double scale);

		static typename QuadricType::Ptr fitQuadric(typename CloudType::ConstPtr cloud, const PointT& pos, NQ& nq, Eigen::Matrix<float,3,3>* localBase = nullptr);
		static float meanCurvature(typename CloudType::Ptr cloud, const NQ& nq, const PointT& pos, const IdxSet& subset = IdxSet());
		static Points getNeighbors(typename CloudType::ConstPtr cloud, const PointT& pos, const NQ& nq);
		static IdxSet getNeighborIndices(typename CloudType::ConstPtr cloud, const PointT& pos, const NQ& nq);

		static float diameter(typename CloudType::ConstPtr cloud);

	protected:
		static Eigen::Vector3f localQuadricParams(typename CloudType::ConstPtr cloud, NeighborQuery<PointT>& nq, const PointT& pos, const Eigen::Matrix3f& localBase);
		static IdxSet cloudIndices(typename CloudType::ConstPtr cloud);

		class NQSearchVisitor : public boost::static_visitor<> {
			public:
				NQSearchVisitor(const PointT& query, typename SearchType::Ptr search, std::vector<int>& indices, std::vector<float>& sqrDists);
				virtual ~NQSearchVisitor();

				void operator()(float param);
				void operator()(unsigned int param);

			protected:
				PointT                    m_query;
				typename SearchType::Ptr  m_search;
				std::vector<int>&         m_indices;
				std::vector<float>&       m_sqrDists;
		};

		template <class Algo>
		class NQSetVisitor : public boost::static_visitor<> {
			public:
				NQSetVisitor(typename SearchType::Ptr search, Algo& algo);
				virtual ~NQSetVisitor();

				void operator()(float param);
				void operator()(unsigned int param);

			protected:
				typename SearchType::Ptr  m_search;
				Algo&                     m_algo;
		};
};


#include "PCLTools.inl"

} // Geometry

#endif /* COMMONSPCLTOOLS_H_ */
